import deepspeed
from ds_arguments import parse_args
from skgutils.configue import Configure
from skgutils.dataset import TokenizedDataset, skg_eval, skg_eval_data_collator
from skgutils.dataset import _pad_tensors_to_max_len, _nested_gather, _pad_across_processes, post_process_function, compute_metrics
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
from transformers.modeling_utils import unwrap_model

from transformers.trainer_utils import has_length
from typing import NamedTuple
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
)
import pdb
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

def get_context(batch, pad_token_id, label_pad_token_id):
    btz = len(batch['input_ids'])
    context_list = []
    input_list = batch['input_ids'].tolist()
    label_list = batch['labels'].tolist()
    for i in range(btz):
        ex_input = input_list[i]
        if pad_token_id in ex_input:
            ex_input = ex_input[:ex_input.index(pad_token_id)]

        ex_label = label_list[i]
        if label_pad_token_id in ex_label:
            ex_label = ex_label[:ex_label.index(label_pad_token_id)]

        context_list += ex_input + ex_label
    return torch.tensor(context_list)


def _left_align_tensor(tokenizer, tensor, device):
    tensor_mask = tensor != tokenizer.pad_token_id
    tensor_mask = tensor_mask.to(device)
    tensor_nonpad = tensor_mask.nonzero(as_tuple=True)
    hh = torch.arange(tensor.size(-1)).expand_as(tensor).to(device)
    output = hh < (tensor_mask).sum(dim=-1, keepdim=True)
    output = output.to(tensor.dtype)
    output_nonzero = output.nonzero(as_tuple=True)
    output.fill_(tokenizer.pad_token_id)
    output[output_nonzero] = tensor[tensor_nonpad]
    return output

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
    # Initialize torch distributed
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = rank % torch.cuda.device_count()

    #* Align different configs.
    deepspeed_config = json.load(open(args.deepspeed_config, 'r', encoding='utf-8'))
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    assert deepspeed_config['gradient_accumulation_steps'] == args.gradient_accumulation_steps
    assert deepspeed_config['train_micro_batch_size_per_gpu'] == args.per_device_train_batch_size
    assert deepspeed_config['gradient_clipping'] == args.grad_clip

    print(f'DEVICE ID: {device_id}', flush=True)

    #* Align different configs.
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
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

    #* Apply PEFT to the T5 model
    if args.add_peft:
        model = modify_transformer(model, config)

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
            # data_cache_root = os.path.join(args.output_dir, 'cache')
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
        #                                  seq2seq_eval_dataset, datatype='train') if seq2seq_eval_dataset else None
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
            logger.info(f'Choose {args.num_shots} training samples...')
            sample_indices = random.sample(range(0, len(raw_train_dataset)), k=args.num_shots)
            raw_train_dataset = raw_train_dataset.select(sample_indices)
            # raw_eval_dataset = raw_eval_dataset.select(sample_indices)

        train_dataset = raw_train_dataset.map(lambda x: t0_preprocess_train(x, tokenizer, template, column_names), batched=True, remove_columns=column_names)
        eval_dataset = raw_eval_dataset.map(lambda x: t0_preprocess_eval(x, tokenizer, template, column_names), batched=True, remove_columns=column_names)
        # test_dataset = raw_test_dataset.map(preprocess_eval, batched=True, remove_columns=column_names)

   #* Log a few random examples:
    if rank <= 0:
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        # for index in random.sample(range(len(eval_dataset)), 3):
        #     logger.info(f"Sample {index} of the evaluation set: {eval_dataset[index]}.")
    
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
    eval_dataloader = DataLoader(eval_dataset, collate_fn=eval_collator, batch_size=args.per_device_eval_batch_size, shuffle=False)
    #test_dataloader = DataLoader(test_dataset, collate_fn=eval_collator, batch_size=args.per_device_eval_batch_size)

    # time.sleep(20)
    if rank <= 0:
        logger.info('Model to CUDA.')
    model = model.to(device_id)
    # time.sleep(20)
    optimizer, trainable_param_names = get_optimizer(model, config)

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

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)
    # model.train()
    model.eval()

    #* Calculate number of parameters 
    num_p = sum([p.numel() for p in model.parameters()])
    tunable_num_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank <= 0: 
        logger.info('Number of parameters: {}'.format(num_p))
        logger.info(f'Number of tunable params: {tunable_num_p}, tunable ratio is {"%.4f"%(tunable_num_p/num_p)}')

    result_table = []
    if args.local_rank <= 0:
        logger.info("********** ICL Running **********")
        logger.info(f"  Num training examples = {len(train_dataset)}")
        logger.info(f"  Num testing examples = {len(eval_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")

   # TODO: Begin Training ------------------------------------
    gen_kwargs = {
        'max_length': args.target_max_length,
        'num_beams': args.generation_num_beams 
    }

    #* Get a fixed batch from the training dataloader
    stage = "icl_eval_all"
    print_output = True
    for step, batch in enumerate(eval_dataloader):
        context_batch = next(iter(train_dataloader))
        context_batch = to_device(context_batch, device_id)
        batch_raw = batch['raw_data']
        eval_batch = to_device(batch, device_id)
        eval_btz = len(eval_batch['input_ids'])


        if "description_input_ids" in batch:
            gen_kwargs["description_input_ids"] = batch["description_input_ids"]
        if "description_attention_mask" in batch:
            gen_kwargs["description_attention_mask"] = batch["description_attention_mask"]
        if "knowledge_input_ids" in batch:
            gen_kwargs["knowledge_input_ids"] = batch["knowledge_input_ids"]
        if "knowledge_attention_mask" in batch:
            gen_kwargs["knowledge_attention_mask"] = batch["knowledge_attention_mask"]
        if "task_ids" in batch:
            gen_kwargs["task_ids"] = batch["task_ids"]

        # print(f'batch keys {context_batch.keys()}', flush=True)

        #* Construct context inputs
        # train_btz = len(context_batch['input_ids'])
        context_array = get_context(context_batch, tokenizer.pad_token_id, -100)
        # context_array = torch.cat(
        #     [
        #         context_batch["input_ids"],
        #         context_batch["labels"],
        #     ],
        #     dim=1,
        # )
        context = context_array.to(device_id)
        # context = _left_align_tensor(tokenizer, context_array, device_id)[:,:]
        # print(f'concat_sen {concat_sen}', flush=True)
        # context = context.view(-1) 
        if rank <= 0 and print_output:
            print(f'Context sentences:\n{tokenizer.decode(context)}', flush=True)

        #* Construct context attention_masks
        context_attn = torch.where(context>0, 1, 0)
        # context_input_attn = context_batch['attention_mask']
        # context_label_attn = torch.where(context_batch['labels']>0, 1, 0)
        # context_attn = torch.cat([context_input_attn, context_label_attn], dim=1)
        # context_attn = context_attn.view(-1)

        #* Construct model inputs
        input_ids = torch.cat(
            [
                context[None, :].expand(eval_btz, -1),
                eval_batch['input_ids'],
            ],
            dim=1,
        )        
        attn_mask = torch.cat([
                context_attn[None, :].expand(eval_btz, -1),
                eval_batch['attention_mask'],
            ], 
            dim=1
        )
        if rank<=0 and print_output:
            print(f'Inputs:\n{tokenizer.batch_decode(input_ids)}', flush=True)

        # input_ids[input_ids == -100] = tokenizer.pad_token_id
        # print(f'input ids {input_ids.size()}', flush=True)

        model = unwrap_model(model)
        with torch.no_grad():
            generated_tokens = model.generate(
                inputs=input_ids, 
                attention_mask=attn_mask,
                **gen_kwargs
                )
        if gen_kwargs.get("max_length") is not None and generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = _pad_tensors_to_max_len(tokenizer, generated_tokens, gen_kwargs["max_length"])
        # if device_id==0: print(f'size of generated_tokens {generated_tokens.size()}', flush=True)

        logits = generated_tokens
        # logits = generated_tokens.view(eval_btz, -1)
        # if device_id==0: print(f'size of reshaped logits {logits.size()}', flush=True)

        logits = nested_detach(logits)
        batch_pred = nested_numpify(logits)
        
        #? Debug--------
        if args.debug:
            if step >= 5:
                break
        #? -------------
        eval_preds = post_process_function(tokenizer, batch_raw, batch_pred, stage, args, device_id, print_output)
        print_output = False

    if device_id == 0 or device_id == 'cuda':    
        with open(f"{args.output_dir}/{stage}.json", "r") as f:
            res = [json.loads(i) for i in f.readlines()]
        all_preds = [ex['prediction'] for ex in res]
        all_labels = res
        final_scores = compute_metrics(evaluator, all_preds, all_labels, section='dev')

    if rank <= 0:
        logger.info(f"********** Final Evaluation **********")
        logger.info(f'FINAL SCORE:')
        for k,v in final_scores.items():
            logger.info(f'{k}: {"%.4f"%v}')
        logger.info(f"{'*'*30}")

        result_table.append({
            "dataset_name": args.dataset_name,
            "score": final_scores,
        })

        with open(os.path.join(args.output_dir, "results.csv"), "w") as f:
            writer = csv.DictWriter(f, fieldnames=result_table[0].keys())
            writer.writeheader()
            writer.writerows(result_table)
    return

def construct_arguments():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # deepspeed.init_distributed()
        deepspeed.init_distributed(dist_backend='nccl', timeout=datetime.timedelta(seconds=1800000))
    args.local_rank = int(os.environ['LOCAL_RANK'])

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
