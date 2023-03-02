from arguments import parse_args
from tools.logger import Logger
from promptsource.templates import DatasetTemplates
from dataset import DataCollatorForMultipleChoice
from models.modify_model import modify_transformer
from myutils.get_optimizer import get_optimizer
from myutils.get_scheduler import get_scheduler
from myutils.Config import Config
from torch.cuda.amp import autocast
import wandb

import os
import sys
import time
import logging
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random, math, sklearn
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm
# import deepspeed
from datasets import load_dataset, load_metric
import evaluate
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    default_data_collator,
    DataCollatorForSeq2Seq,
    AdamW,
    SchedulerType,
    #get_scheduler,
    set_seed,
)
from transformers.file_utils import PaddingStrategy
torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

def to_json(config):
    return json.dumps(config, indent=4, sort_keys=True)

def to_dict(config):
    for k, v in config.items():
        config[k] = repr(v)
    return config

def save(model, tokenizer, save_folder):
    model.save_pretrained(save_folder, state_dict=model.module.state_dict())
    tokenizer.save_pretrained(save_folder)

def get_dataset(args):
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,)
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,)

    return raw_datasets


def train(args):
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

    #* Align different configs.
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    config = Config(args.config_files, args.kwargs) # most for PEFT modules, optimizer and lr scheduler
    config.exp_dir = args.output_dir
    print('CONFIGS:',config.to_json(), flush=True)

    if args.local_rank <= 0:
        run_config = vars(args)
        filename = os.path.join(args.output_dir, 'myargs.json')
        with open(filename, "w") as fout:
            arg_dict = to_dict(run_config)
            fout.write(to_json(arg_dict))
            fout.write('\n')
        wandb.init(project=config.exp_name, config=run_config, dir=args.wandb_log)
        # wandb.init(project=args.wandb_log)

   #! Loading Model, Optimizer and Tokenizer
    #* Loading Model
    if args.local_rank <= 0:
        logger.info(f'Begin Loading the model...')
    assert args.model_name_or_path, "Need to specify the model name or path!"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path, 
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=model_config,
    )
    time.sleep(20)
    if args.local_rank <= 0:
        logger.info('Model to CUDA.')
    model = model.to(device)

    #* Loading Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
    )

    # model.resize_token_embeddings(len(tokenizer))

    #* Apply PEFT to the T5 model
    if args.add_peft:
        model = modify_transformer(model, config)

        optimizer, trainable_param_names = get_optimizer(model, config)
        print(f'Trainable parameters are:\n {trainable_param_names}', flush=True)
    else:
        optimizer_grouped_parameters = prepare_optimizer_parameters(config, model)
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr)    
    
    # model, optimizer, tokenizer, lr_scheduler = prepare_model_optimizer(args)

    #* Prepare the template
    prompts = DatasetTemplates(
        f"{args.dataset_name}"
        if args.dataset_config_name is None
        else f"{args.dataset_name}/{args.dataset_config_name}"
        )
    template = prompts[args.template_name]

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_train(examples):
        bs = len(examples[column_names[0]])

        input_texts = []
        target_texts = []
        for i in range(bs):
            ex = {k: examples[k][i] for k in column_names}
            input, target = template.apply(ex)
            # ex_answer_choices = template.get_answer_choices_list(ex)
            # assert target in ex_answer_choices
            # print(f'target {target}\n {ex_answer_choices}', flush=True)
            input_texts.append(input)
            target_texts.append(target)

        model_inputs = tokenizer(
            input_texts,
            padding=padding,
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=True,
        )

        # with tokenizer.as_target_tokenizer():
        tokenized_targets = tokenizer(
            target_texts,
            padding=padding,
            max_length=args.target_max_length,
            truncation=True,
            add_special_tokens=True,
        )
        # print(f'tgt {target_texts}, {tokenized_targets}', flush=True)
        model_inputs['labels'] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in targets]
            for targets in tokenized_targets["input_ids"]
        ]
        return model_inputs

    def preprocess_eval(examples):
        bs = len(examples[column_names[0]])

        input_texts = []
        target_texts = []
        answer_choices_texts = []
        for i in range(bs):
            ex = {
                k: examples[k][i]
                for k in column_names
            }
            input, target = template.apply(ex)
            ex_answer_choices = template.get_answer_choices_list(ex)
            try:
                assert target in ex_answer_choices
            except:
                print(f'target {target}\n {ex_answer_choices}', flush=True)
                continue
            
            input_texts.append(input)
            target_texts.append(target)
            answer_choices_texts.append(ex_answer_choices)

        bs = len(input_texts)

        tokenized_inputs = tokenizer(
            input_texts,
            padding=padding,
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=True,
            # return_tensors="pt",
        )
        tokenized_targets = [
            tokenizer(
                ans_choi,
                padding=True,
                max_length=args.target_max_length,
                truncation=True,
                # return_tensors="pt",
            )
            for ans_choi in answer_choices_texts
        ]

        features = {
            k: [
                [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
                for idx, elem in enumerate(v)
            ]
            for k, v in tokenized_inputs.items()
        }

        features["labels"] = [
            tokenized_targets[idx]["input_ids"]
            for idx in range(bs)
        ]
        features["labels_attention_mask"] = [
            tokenized_targets[idx]["attention_mask"]
            for idx in range(bs)
        ]
        features["targets"] = [
            answer_choices_texts[idx].index(t)
            for idx, t in enumerate(target_texts)
        ]

        return features

    def eval(ddloader):
        model.eval()
        eval_metric = evaluate.load('accuracy')
        for batch in ddloader:
            batch = to_device(batch, device)
            model_inputs = {k: batch[k] for k in ["input_ids", "attention_mask", "labels"]}
            # inp_ids = model_inputs['input_ids']
            # attn_mask = model_inputs['attention_mask']
            # labels = model_inputs['labels']

            with torch.no_grad():
                outs = model(**model_inputs)
                logits = outs.logits

            masked_log_probs = batch["labels_attention_mask"].unsqueeze(-1) * torch.log_softmax(logits, dim=-1)
            seq_token_log_probs = torch.gather(masked_log_probs, -1, batch["labels"].unsqueeze(-1))
            seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
            seq_log_prob = seq_log_prob.view(batch["targets"].size(0), -1) #TODO(Victor): this reshapes works based on the assumption that all examples have the same number of choices. the pre-processing doesn't make this assumption.
            predictions = seq_log_prob.argmax(dim=-1)

            eval_metric.add_batch(predictions=predictions, references=batch["targets"],)
        
        score = eval_metric.compute()["accuracy"]
        return score

   #* Loading Dataset
    raw_datasets = get_dataset(args)
    raw_train_dataset, raw_eval_dataset = raw_datasets['train'], raw_datasets['validation']

    # Trim a number of evaluation examples
    if args.debug:
        raw_train_dataset = raw_train_dataset.select(range(min(100, len(raw_train_dataset))))
        raw_eval_dataset = raw_eval_dataset.select(range(min(100, len(raw_eval_dataset))))
        # raw_test_dataset = raw_test_dataset.select(range(min(100, len(raw_test_dataset))))

    column_names = raw_eval_dataset.column_names

    if args.num_shots is not None:
        sample_indices = random.sample(range(0, len(raw_train_dataset)), k=args.num_shots)
        raw_train_dataset = raw_train_dataset.select(sample_indices)
    train_dataset = raw_train_dataset.map(preprocess_train, batched=True, remove_columns=column_names)
    eval_dataset = raw_eval_dataset.map(preprocess_eval, batched=True, remove_columns=column_names)
    # test_dataset = raw_test_dataset.map(preprocess_eval, batched=True, remove_columns=column_names)

   #* Log a few random examples:
    if args.local_rank <= 0:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        for index in random.sample(range(len(eval_dataset)), 3):
            logger.info(f"Sample {index} of the evaluation set: {eval_dataset[index]}.")
        # for index in random.sample(range(len(test_dataset)), 3):
        #     logger.info(f"Sample {index} of the evaluation set: {test_dataset[index]}.")
    
        #* Prepare writer
        logger.info(f'Prepare tensorboard writer...')
        train_writer = SummaryWriter(os.path.join(args.tbdir, 'train'), flush_secs=10)
        valid_writer = SummaryWriter(os.path.join(args.tbdir, 'eval'))

   #* DataLoaders Creation:
    train_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=None)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=train_collator,batch_size=args.per_device_train_batch_size)

    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        eval_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        eval_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=None
        )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=eval_collator, batch_size=args.per_device_eval_batch_size)
    # test_dataloader = DataLoader(test_dataset, collate_fn=eval_collator, batch_size=args.per_device_eval_batch_size)

   #* Setup LR scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    #lr_scheduler = get_scheduler(
            #name=args.lr_scheduler_type,
            #optimizer=optimizer,
            #num_warmup_steps=args.num_warmup_steps,
            #num_training_steps=args.max_train_steps,)
    config.num_steps = int(args.max_train_steps) # align args hyparam with config 
    lr_scheduler = get_scheduler(
        optimizer=optimizer,
        config=config,
    )

    model.train()

    #* Calculate number of parameters 
    num_p = sum([p.numel() for p in model.parameters()])
    tunable_num_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.local_rank <= 0: 
        logger.info('Number of parameters: {}'.format(num_p))
        logger.info(f'Number of tunable params: {tunable_num_p}, tunable ratio is {"%.4f"%(tunable_num_p/num_p)}')

    #* Metrics
    # metric = load_metric("accuracy")
    metric = evaluate.load('accuracy')
    result_table = []

    if args.local_rank <= 0:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

   # TODO: Begin Training ------------------------------------
    global_steps = 0

    if args.debug_infer:
        input_seq = tokenizer(
            ["Applies a linear transformation to the incoming data."],
            return_tensors="pt",
        )
        print(f'Test input {input_seq}', flush=True)
        target_seq = tokenizer(
            ["Parameters: in_features - size of each input sample. out_features - size of each output sample."],
            return_tensors="pt",
        )
        print(f'Test tgt {target_seq}', flush=True)
        with torch.no_grad():
            new_outputs = model(
                input_ids=input_seq.input_ids.cuda(),
                decoder_input_ids=target_seq.input_ids[:, :-1].cuda(),
                labels=target_seq.input_ids[:, 1:].cuda(),
            )
        print(f'Test example outputs: {new_outputs}', flush=True)
        return

    for epoch in range(args.num_train_epochs):
        if args.local_rank <= 0:
            logger.info(f'{"*"*20} EPOCH {epoch} {"*"*20}')
        model.train()

        for step, batch in enumerate(tqdm(train_dataloader, desc=f'EPOCH {epoch}')):
            model.train()
            # print(f"batch size {len(batch['input_ids'])}", flush=True)
            # print(f'batch {batch}', flush=True)

            batch = to_device(batch, device)
            # outputs = model(**batch)
            if args.use_a100:
                with autocast(dtype=torch.bfloat16):
                    outputs = model(input_ids=batch['input_ids'],labels=batch['labels'])
                    loss = outputs.loss
                    loss.backward() # just for a test
            else:
                outputs = model(input_ids=batch['input_ids'],labels=batch['labels'])

            # print(f'LOGITS: {outputs.logits}; LOSS: {outputs.loss}',flush=True)
            # print('OUTPUTS',outputs,flush=True)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            # return
            loss.backward()

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                global_steps += 1
                loss = loss.item()
                lr_scheduler.step()
                if args.local_rank <= 0:
                    train_writer.add_scalar('train/loss', loss, global_steps)
                    wandb.log({"loss":loss}, step=global_steps)
                    lr = lr_scheduler.get_last_lr()[0]
                    train_writer.add_scalar('lr',lr, global_steps)
                    wandb.log({"lr":lr}, step=global_steps)
                    if step % 10 ==0:
                        logger.info(f'TRAIN--> EPOCH {epoch}; STEP {step}; LR {lr}; LOSS {loss}')
            if global_steps >= args.max_train_steps:
                break

            #* evaluate during one epoch...
            if global_steps % args.eval_step == 0 or step == len(train_dataloader)-1:     
                eval_score = eval(eval_dataloader)
                if args.local_rank <= 0: 
                    logger.info(f'EVAL--> EPOCH {epoch}; GLOBAL STEP {global_steps}; STEP {step}; EVAL_SCORE {eval_score}')
                    valid_writer.add_scalar('eval/score', eval_score, global_steps)
                    wandb.log({"score":eval_score},step=global_steps)

       # TODO: Evaluate the model after training per epoch----------------------
        # jjj = 0
        # print(f'for {jjj} batch: inp_ids {inp_ids[0], inp_ids.shape}')
        # print(f'for {jjj} batch: attn_mask {attn_mask[0], attn_mask.shape}')
        # print(f'for {jjj} batch: label {labels[0], labels.shape}')
        score = eval(eval_dataloader)       
        if args.local_rank <= 0: 
            logger.info(f'{"*"*10} EPOCH {epoch}; GLOBAL STEP {global_steps}; SCORE: {score} {"*"*10}')
            wandb.log({"score":score}, step=global_steps)

        result_table.append({
            "dataset_name": args.dataset_name,
            "dataset_config_name": args.dataset_config_name,
            "template_name": args.template_name,
            "epoch": epoch,
            "step": global_steps,
            "metric": 'accuracy',
            "score": score,
        })

       # TODO: Save model checkpoints ---------------------------------
        if args.output_dir is not None:
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir)
            
            if args.local_rank <= 0:
                logger.info(f'Begin saving model to {args.output_dir}')
                model_to_save = model.module if hasattr(model, 'module') else model
                CONFIG_NAME = "config.json"
                WEIGHTS_NAME = "pytorch_model.bin"
                output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                torch.save(model_to_save, output_model_file)
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(args.output_dir)

    if args.local_rank <= 0:
        logger.info(f'Finish Training the model for {epoch} epochs.') 

    # if args.wandb_log:
    #     wandb.finish()

def construct_arguments():
    args = parse_args()
    os.environ["WANDB_API_KEY"] = '9bb23cf3c9acc6172c9d16d4045b0262f78376c6'
    # os.environ["WANDB_MODE"] = "offline"
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = args.master_port

    print("Args = {}".format(args))

    # Setting all the seeds so that the task is random but same accross processes
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f'Finish arguments construction!')
    return args

def prepare_optimizer_parameters(config, model):

    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    return optimizer_grouped_parameters

if __name__ == "__main__":
    start = time.time()
    #* Prepare Logger
    logger = Logger(cuda=torch.cuda.is_available())
    logger.info(f'Constructing arguments...')

    args = construct_arguments()

    train(args)

    elapsed = time.time() - start
    if args.local_rank <= 0: 
        logger.info(f"Elapsed time: {elapsed} seconds")
        logger.info(f'Finish Training! Congrats!!!')
