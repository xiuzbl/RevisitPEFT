from arguments import parse_args
from skgutils.configue import Configure
from skgutils.dataset import TokenizedDataset, skg_eval, skg_eval_data_collator
from skgutils.trainer import EvaluateFriendlySeq2SeqTrainer
from skgutils.tool import get_constructor, get_evaluator
from tools.logger import Logger
from promptsource.templates import DatasetTemplates
from t0dataset import DataCollatorForMultipleChoice, get_input_target, t0_preprocess_train, t0_preprocess_eval, t0_eval
from models.modify_model import modify_transformer
from myutils.get_optimizer import get_optimizer
from myutils.get_scheduler import get_scheduler
from myutils.Config import Config
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
import wandb
import re
import copy 
from collections import defaultdict

import os
import sys
import time
import logging
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random, math, sklearn
import json, csv
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from datasets import load_dataset, load_metric
# import evaluate
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

def to_dict(args):
    print(f'args {args}', flush=True)
    config = vars(copy.deepcopy(args))
    for k, v in config.items():
        config[k] = repr(v)
    print(f'new args {args}', flush=True)
    return config

def print_grads(model, config, global_step, outdir):
    def param_name_to_group_name(param_name):
        if False:
            return ".".join(param_name.split(".")[:3])
            # only needed when the model has many trainable parameters, disabled in our expeirments
        else:
            return "."

    outfile = os.path.join(outdir, f'grads.txt')
    param_groups = defaultdict(lambda: {"params": []})
    adapter_params = ".*layer_norm.*|.*adapter.*"

    trainable_param_names = set()
    grad_norm_list = []
    for (param_name, param) in model.named_parameters():
        # with deepspeed.zero.GatheredParameters(param, modifier_rank=None):
        if re.fullmatch(adapter_params, param_name):
            param_groups[param_name_to_group_name(param_name)]["params"].append(param)
            trainable_param_names.add(param_name)
            if param.grad is not None:
                grad_norm_list.append(param.grad.norm().item())
        # else:
        #     param.requires_grad = False
    if len(grad_norm_list):
        with open(outfile, 'a') as fw:
            print(f'Global step: {global_step}', file=fw)
            print(f'Trainable params gradient list {grad_norm_list}', file=fw)
            print(f'Average gradient norm {sum(grad_norm_list)/len(grad_norm_list)}', file=fw)
    return

def record_grads(tb, model, step):
    param_groups = defaultdict(lambda: {"params": []})
    adapter_params = ".*adapter.*"
    lora_params = '.*lora_[ab].*'
    layer_norm_params = '.*layer_norm.*'

    trainable_param_names = set()
    grad_norm_list = []
    for (param_name, param) in model.named_parameters():
        # if re.fullmatch(adapter_params, param_name) or re.fullmatch(lora_params, param_name) or re.fullmatch(layer_norm_params, param_name):
        if param.grad is not None:
            # grad_value = param.grad.norm().item()
            grad_value = param.grad
            if 'bias' in param_name:
                bias = param
                tb.add_histogram(param_name[:-5]+'/bias', bias, step)
                tb.add_histogram(param_name[:-5]+'/bias_grad', grad_value, step)
            if 'weight' in param_name:
                # print(param, flush=True)
                weight = param
                tb.add_histogram(param_name[:-7]+'/weight', weight, step)
                tb.add_histogram(param_name[:-7]+'/weight_grad', grad_value, step)
                tb.add_scalar(param_name[:-7]+'/weight_grad_scalar', grad_value.norm(), step)
    return 

def close_fnn_grads(model, other_params_list):
    for (param_name, param) in model.named_parameters():
        for other_param in other_params_list:
            if re.fullmatch(other_param, param_name):
                param.requires_grad = False
    return

def calculate_tunable_ratio(model, logger):
    #* Calculate number of parameters 
    num_p = sum([p.numel() for p in model.parameters()])
    tunable_num_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_list = []
    for (param_name, param) in model.named_parameters():
        if param.requires_grad:
            trainable_list.append(param_name)
    print(f'Total trainable parameter names {len(trainable_list)}', flush=True)
    print(f'Trainable list {trainable_list}', flush=True)
    logger.info('Number of parameters: {}'.format(num_p))
    logger.info(f'Number of tunable params: {tunable_num_p}, tunable ratio is {"%.4f"%(tunable_num_p/num_p)}')
    return

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

    #* Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    # args.local_rank = int(os.environ['LOCAL_RANK'])
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=180000))
        world_size = torch.distributed.get_world_size()

        rank = dist.get_rank()
        device_id = rank % torch.cuda.device_count()
    else:
        world_size = 1
        rank = args.local_rank
        device_id = 'cuda'
    print(f'WORLD SIZE: {world_size}; RANK: {rank}; DEVICE ID: {device_id}', flush=True)

    #* Align different configs.
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    # print(f'config vocab {model_config.vocab_size}', flush=True)
    # model_config = AutoConfig.from_pretrained('t5-base')
    config = Config(args.config_files, args.kwargs) # most for PEFT modules, optimizer and lr scheduler
    config.exp_dir = args.output_dir
    config.warmup_ratio = float(config.warmup_ratio)
    config.lr = float(config.lr)
    config.adapter_reduction_factor = int(config.adapter_reduction_factor)

    if rank <= 0:
        print('CONFIGS:',config.to_json(), flush=True)

    if args.local_rank <= 0:
        filename = os.path.join(args.output_dir, 'myargs.json')
        with open(filename, "w") as fout:
            arg_dict = to_dict(args)
            fout.write(to_json(arg_dict))
            fout.write('\n')
 
   #! Loading Model, Optimizer and Tokenizer
    #* Loading Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
    )

    #* Loading Model
    if rank <= 0:
        logger.info(f'Begin Loading the model...')
    assert args.model_name_or_path, "Need to specify the model name or path!"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path, 
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=model_config,
    )
    # model = AutoModelForSeq2SeqLM.from_config(model_config) #? random initialize the model

    #* Apply PEFT to the T5 model
    if args.add_peft:
        model = modify_transformer(model, config)

    if rank <= 0: 
        model_arch_file = os.path.join(args.output_dir, 'model_architecture.txt')
        with open(model_arch_file,'w') as f:
            for i in model.named_modules():
                print(i, file=f)
            print('\n',file=f)
            print('*'*100, file=f)
            print('*'*100, file=f)
            print(model, file=f)

    if not args.skg_task:
        #* Prepare the template
        prompts = DatasetTemplates(
            f"{args.dataset_name}"
            if args.dataset_config_name is None
            else f"{args.dataset_name}/{args.dataset_config_name}"
            )
        template = prompts[args.template_name]

    padding = "max_length" if args.pad_to_max_length else False

   #* Loading Dataset
    if args.skg_task:
        #? SKG dataset process 
        logger.info(f'Dealing with SKG tasks-------------------------------')
        skgargs = Configure.Get(args.skg_cfg)
        logger.info(f'Length of tokenizer: {len(tokenizer)}', flush=True)

        if skgargs.special_tokens:
            tokenizer.add_tokens([v for k,v in skgargs.special_tokens])
            logger.info(f'Length of tokenizer new: {len(tokenizer)}', flush=True)
            model.resize_token_embeddings(len(tokenizer))

        meta_tuning_data = {}
        for task, arg_path in skgargs.arg_paths:
            task_args = Configure.Get(arg_path)
            # task_args.bert = args.bert
            # print('task_args.bert.location:', task_args.bert.location)
            task_raw_datasets_split = load_dataset(path=task_args.dataset.loader_path,
                                                   cache_dir=task_args.dataset.data_store_path)
            data_cache_root = os.path.join(task_args.dataset.data_store_path, 'cache')
            os.makedirs(data_cache_root, exist_ok=True)
            task_seq2seq_dataset_split = get_constructor(task_args.seq2seq.constructor)(task_args).to_seq2seq(task_raw_datasets_split, data_cache_root)

            meta_tuning_data[arg_path] = task_seq2seq_dataset_split

        seq2seq_dataset_split = get_constructor(skgargs.seq2seq.constructor)(skgargs).to_seq2seq(meta_tuning_data)

        evaluator = get_evaluator(skgargs.evaluate.tool)(skgargs)

        seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = None, None, None
        if len(seq2seq_dataset_split) == 2:
            seq2seq_train_dataset, seq2seq_eval_dataset = seq2seq_dataset_split
        elif len(seq2seq_dataset_split) == 3:
            seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = seq2seq_dataset_split
        else:
            raise ValueError("Other split not support yet.")

        # We wrap the "string" seq2seq data into "tokenized tensor".
        # if args.debug:
        #     train_dataset = TokenizedDataset(skgargs, args, tokenizer,
        #                                  seq2seq_eval_dataset) if seq2seq_eval_dataset else None
        # else:
        train_dataset = TokenizedDataset(skgargs, args, tokenizer,
                                         seq2seq_train_dataset, datatype='train') if seq2seq_train_dataset else None
        eval_dataset = TokenizedDataset(skgargs, args, tokenizer,
                                        seq2seq_eval_dataset, datatype='eval') if seq2seq_eval_dataset else None
        test_dataset = TokenizedDataset(skgargs, args, tokenizer,
                                    seq2seq_test_dataset, datatype='test') if seq2seq_test_dataset else None
    else:
        #? T0 dataset process
        raw_datasets = get_dataset(args)
        if args.debug:
            if rank <= 0: logger.info(f'Debugging......')
            raw_train_dataset, raw_eval_dataset = raw_datasets['validation'], raw_datasets['validation']
        else:
            raw_train_dataset, raw_eval_dataset = raw_datasets['train'], raw_datasets['validation']
        if rank <= 0: logger.info(f'Number of Raw Train-set {raw_train_dataset}; Raw Eval-set {raw_eval_dataset}')

        # Trim a number of evaluation examples
        if args.debug:
            raw_train_dataset = raw_train_dataset.select(range(min(100, len(raw_train_dataset))))
            raw_eval_dataset = raw_eval_dataset.select(range(min(100, len(raw_eval_dataset))))
            # raw_test_dataset = raw_test_dataset.select(range(min(100, len(raw_test_dataset))))

        column_names = raw_eval_dataset.column_names

        if args.num_shots is not None:
            num_shots = min(len(raw_train_dataset), args.num_shots)
            logger.info(f'Choose {num_shots} training samples...')
            sample_indices = random.sample(range(0, len(raw_train_dataset)), k=num_shots)
            raw_train_dataset = raw_train_dataset.select(sample_indices)
            # raw_eval_dataset = raw_eval_dataset.select(sample_indices)

        train_dataset = raw_train_dataset.map(lambda x: t0_preprocess_train(x, tokenizer, template, column_names, args), batched=True, remove_columns=column_names)
        eval_dataset = raw_eval_dataset.map(lambda x: t0_preprocess_eval(x, tokenizer, template, column_names, args), batched=True, remove_columns=column_names)
        # test_dataset = raw_test_dataset.map(preprocess_eval, batched=True, remove_columns=column_names)

   #* Log a few random examples:
    if rank <= 0:
        # for index in random.sample(range(len(train_dataset)), 3):
        for index in range(3):
            logger.info(f"Sample {index} of the training set:\n {train_dataset[index]}.")
        # for index in random.sample(range(len(eval_dataset)), 3):
        for index in range(3):
            logger.info(f"Sample {index} of the evaluation set:\n {eval_dataset[index]}.")
        # for index in random.sample(range(len(test_dataset)), 3):
        #     logger.info(f"Sample {index} of the evaluation set: {test_dataset[index]}.")
    
        #* Prepare writer
        logger.info(f'Prepare tensorboard writer...')
        train_writer = SummaryWriter(os.path.join(args.tbdir, 'train'), flush_secs=10)
        valid_writer = SummaryWriter(os.path.join(args.tbdir, 'eval'))

   #* DataLoaders Creation:
    train_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=None)
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        train_dataloader = DataLoader(train_dataset, collate_fn=train_collator,batch_size=args.per_device_train_batch_size, sampler=train_sampler)
    else:
        train_dataloader = DataLoader(train_dataset, collate_fn=train_collator, shuffle=True, batch_size=args.per_device_train_batch_size)

    if args.pad_to_max_length or args.skg_task:
        # If padding was already done ot max length, we use the default data collator that will just convert everything to tensors.
        # eval_collator = default_data_collator
        eval_collator = skg_eval_data_collator
    else: # T0 task data collator
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        eval_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=None
        )
    if world_size > 1:
        # eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
        # eval_dataloader = DataLoader(eval_dataset, collate_fn=eval_collator, batch_size=args.per_device_eval_batch_size, sampler=eval_sampler)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=eval_collator, batch_size=args.per_device_eval_batch_size, shuffle=False)
    else:
        eval_dataloader = DataLoader(eval_dataset, collate_fn=eval_collator, shuffle=False, batch_size=args.per_device_eval_batch_size)
    #test_dataloader = DataLoader(test_dataset, collate_fn=eval_collator, batch_size=args.per_device_eval_batch_size)

    #* Make model distributed.
    # time.sleep(20)
    if rank <= 0:
        logger.info('Model to CUDA.')
    model = model.to(device_id)
    # time.sleep(20)
    if rank <= 0 and world_size > 1:
        logger.info(f'Begin DDP the model...')
    if world_size > 1:
        model = DDP(model, 
                device_ids=[device_id], 
                output_device=device_id, 
                gradient_as_bucket_view=True,
                find_unused_parameters=True,
        )

    # other_params_list = ['.*DenseReluDense.wi.*']
    # other_params_list = ['.*11.layer.*.DenseReluDense.*', '.*10.layer.*.DenseReluDense.*']
    # other_params_list = None
    other_params_list = args.other_params_list
    if args.reset_trainable_params:
        config.trainable_param_names = other_params_list
    # optimizer, trainable_param_names = get_optimizer(model, config, other_param_names=None)
    optimizer, trainable_param_names = get_optimizer(model, config, other_param_names=other_params_list)

    if rank <= 0:
        print(f'Trainable parameters has {len(trainable_param_names)} parameters\nThey are:\n {trainable_param_names}', flush=True)
    # model, optimizer, tokenizer, lr_scheduler = prepare_model_optimizer(args)

   #* Setup LR scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    config.num_steps = int(args.max_train_steps) # align args hyparam with config 
    lr_scheduler = get_scheduler(
        optimizer=optimizer,
        config=config,
    )

    model.train()

    #* Calculate number of parameters 
    if rank <= 0:
        calculate_tunable_ratio(model, logger)

    #* Metrics
    # metric = load_metric("accuracy")
    # metric = evaluate.load('accuracy')
    result_table = []

    if rank <= 0:
        logger.info("********** Running training **********")
        logger.info(f"  Num training examples = {len(train_dataset)}")
        logger.info(f"  Num testing examples = {len(eval_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(f"  Number Update steps per epoch = {num_update_steps_per_epoch}")

   # TODO: Begin Training ------------------------------------
    global_steps = 0

   #? Debugging for inference ......
    if args.debug_infer and rank <= 0:
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

   #? Training .......
    if args.do_train:
        for epoch in range(args.num_train_epochs):
            if rank <= 0:
                logger.info(f'{"*"*20} EPOCH {epoch} {"*"*20}')
            model.train()

            #for step, batch in enumerate(tqdm(train_dataloader, desc=f'EPOCH {epoch}')):
            for step, batch in enumerate(train_dataloader):
                model.train()
                # print(f"batch size {len(batch['input_ids'])}", flush=True)
                # print(f'batch {batch}', flush=True)

                batch = to_device(batch, device_id)
                if args.use_a100:
                    with autocast(dtype=torch.bfloat16):
                        outputs = model(input_ids=batch['input_ids'],labels=batch['labels'])
                else:
                    # outputs = model(**batch)
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])

                # print(f'LOGITS: {outputs.logits}; LOSS: {outputs.loss}',flush=True)
                # print('OUTPUTS',outputs,flush=True)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                # return
                loss.backward()

                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    if rank <= 0:
                        print_grads(model, config, global_steps, args.output_dir)
                        if args.tb_grads:
                            record_grads(train_writer, model, global_steps) #? Record adapter gradient norm
                    optimizer.step()
                    global_steps += 1
                    loss = loss.item()
                    lr_scheduler.step()
                    if rank <= 0:
                        train_writer.add_scalar('train/loss', loss, global_steps)
                        lr = lr_scheduler.get_last_lr()[0]
                        train_writer.add_scalar('lr',lr, global_steps)
                        if step % 10 ==0:
                            logger.info(f'train--> epoch {epoch}; step {step}; lr {lr}; loss {loss}')
                    optimizer.zero_grad()
                if global_steps >= args.max_train_steps:
                    break

                if args.early_stop_steps != -1:
                    if global_steps >= args.early_stop_steps: break

                #* evaluate during one epoch...
                if global_steps % args.eval_step == 0 and step % args.gradient_accumulation_steps==0 and args.do_eval:     
                    if rank <= 0 or world_size == 1: 
                        logger.info(f"***** Running Evaluation *****")
                        # dist.barrier()
                    if args.skg_task:
                        eval_score = skg_eval(model, tokenizer, eval_dataloader, seq2seq_eval_dataset, 
                                             device=device_id, args=args, epoch=epoch, evaluator=evaluator, 
                                             num_eval_samples=len(eval_dataset), step=global_steps, eval_part=True)
                    else:
                        eval_score = t0_eval(model, eval_dataloader, device=device_id)
                    if world_size > 1: 
                        dist.barrier()
                    if (rank <= 0 or world_size == 1) and eval_score is not None: 
                        logger.info(f"EVALUATION ==> {'*'*30}")
                        logger.info(f'EPOCH: {epoch}')
                        logger.info(f'GLOBAL_STEP: {global_steps}')
                        logger.info(f'STEP: {step}')
                        logger.info(f'PART EVAL_SCORE:')
                        if type(eval_score)==dict:
                            for k,v in eval_score.items():
                                logger.info(f'{k}: {"%.4f"%v}')
                                valid_writer.add_scalar('eval/'+k, v, global_steps)
                        else:
                            valid_writer.add_scalar('eval/score', eval_score, global_steps)
                        logger.info(f"{'*'*30}")

                        result_table.append({
                            "epoch": epoch,
                            "global_step": global_steps,
                            "score": eval_score,
                            "dataset_name": args.dataset_name,
                            "dataset_config_name": args.dataset_config_name,
                        })

                if args.debug:
                    break

            if args.early_stop_steps != -1:
                if global_steps >= args.early_stop_steps: break

           # TODO: Evaluate the model after training per epoch----------------------
            if args.skg_task and args.do_eval:
                eval_part = True if args.debug else False
                score = skg_eval(model, tokenizer, eval_dataloader, seq2seq_eval_dataset, 
                                 device=device_id, args=args, epoch=epoch, evaluator=evaluator, 
                                 num_eval_samples=len(eval_dataset), step=global_steps, eval_part=eval_part)
            elif args.do_eval:
                score = t0_eval(model, eval_dataloader, device=device_id)

            if rank <= 0 and args.do_eval and score is not None: 
                logger.info(f"***** EPOCH-{epoch} Evaluation *****")
                logger.info(f'GLOBAL STEP: {global_steps}\nSCORE: {score}')
                if type(score)==dict:
                    for k,v in score.items():
                        valid_writer.add_scalar('eval/'+k, v, global_steps)
                else:
                    valid_writer.add_scalar('eval/score', score, global_steps)
                logger.info(f"{'*'*30}")

                # result_table.append({
                #     "epoch": epoch,
                #     "global_step": global_steps,
                #     "score": score,
                #     "dataset_name": args.dataset_name,
                #     "dataset_config_name": args.dataset_config_name,
                # })

            # TODO: Save model checkpoints ---------------------------------
            if args.output_dir is not None and epoch % 5 == 0:
                save_folder = os.path.join(args.output_dir,'ckp_epoch'+str(epoch))            
                os.makedirs(args.output_dir, exist_ok=True)
                os.makedirs(save_folder, exist_ok=True)
                if rank <= 0:
                    logger.info(f'Begin saving model to {save_folder}')
                    model_to_save = model.module if hasattr(model, 'module') else model
                    CONFIG_NAME = "config.json"
                    WEIGHTS_NAME = "pytorch_model.bin"
                    output_model_file = os.path.join(save_folder, WEIGHTS_NAME)
                    output_config_file = os.path.join(save_folder, CONFIG_NAME)
                    torch.save(model_to_save, output_model_file)
                    output_config_file = os.path.join(save_folder, CONFIG_NAME)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_vocabulary(save_folder)
                    # peft_path = os.path.join(save_folder, 'peft_module.pt')

            if args.close_fnn_epochs != -1 and epoch >= args.close_fnn_epochs:
                if epoch == args.close_fnn_epochs:
                    # print(f'new model config vocab {model_config.vocab_size}', flush=True)
                    close_fnn_grads(model, other_params_list)
                    if args.recover_weights:
                        # ori_model = torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin'))
                        ori_config = AutoConfig.from_pretrained(args.model_name_or_path)
                        ori_model = AutoModelForSeq2SeqLM.from_pretrained(
                            args.model_name_or_path, 
                            from_tf=bool(".ckpt" in args.model_name_or_path),
                            config=ori_config,
                        )
                        ori_model.resize_token_embeddings(len(tokenizer))
 
                        if rank<=0: logger.info(f'Begin recovering the original pretrained model weights...')
                        # dict_param = dict(model.named_parameters())
                        # print(f'param keys {dict_param.keys()}', flush=True)
                        dict_module_param = dict(model.module.named_parameters())
                        # print(f'param module keys {dict_module_param.keys()}', flush=True)
                        for ori_name, ori_param in ori_model.named_parameters():
                            if ori_name in dict_module_param:
                                # print(ori_name, flush=True)
                                dict_module_param[ori_name].data.copy_(ori_param.data) 
                            else:
                                print(f'Not exist in the current model {ori_name}', flush=True)
                        from collections import OrderedDict
                        new_state_dict = OrderedDict()
                        for k, v in dict_module_param.items():
                            if 'module' not in k: # load weights to DDP model
                                k = 'module.' + k
                            new_state_dict[k] = v
                        if rank <= 0:
                            print(f'{model.state_dict().keys()}', flush=True)
                            print(f'{new_state_dict.keys()}', flush=True)
                        model.load_state_dict(new_state_dict, strict=False)
                        if rank<=0: logger.info(f'Finish recovering the original pretrained model weights!')
                if rank <= 0:
                    logger.info(f'Changing tunable ratio...')
                    calculate_tunable_ratio(model, logger) # check whether freeze fnn layers

       # TODO: Save model checkpoints ---------------------------------
        if args.output_dir is not None and not args.debug:
            if rank <= 0:
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
                # peft_path = os.path.join(args.output_dir, 'peft_module.pt')

    if args.local_rank <= 0 and not args.debug:
        logger.info(f'Finish Training the model.') 
        with open(os.path.join(args.output_dir, "results.csv"), "w") as f:
            writer = csv.DictWriter(f, fieldnames=result_table[0].keys())
            writer.writeheader()
            writer.writerows(result_table)

def construct_arguments():
    args = parse_args()
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = args.master_port

    # Setting the distributed variables
    print("Args = {}".format(args))

    # Setting all the seeds so that the task is random but same accross processes
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.distributed.barrier()
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
