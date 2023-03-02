import os
import torch
import json
import torch.distributed as dist
from transformers.modeling_utils import unwrap_model
from collections.abc import Mapping
from torch import nn
from torch.utils.data import Dataset
from transformers.trainer_utils import has_length
from typing import NamedTuple
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
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

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


class EvalPrediction(NamedTuple):
    predictions: List[str]
    items: List[dict]

class TokenizedDataset(Dataset):
    # TODO: A unified structure-representation.
    def __init__(self, args, training_args, tokenizer, seq2seq_dataset, datatype):
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.seq2seq_dataset = seq2seq_dataset
        self.datatype=datatype

        self.conv_sep = " || "

    def __getitem__(self, index):
        raw_item = self.seq2seq_dataset[index]
        # print(f'skgargs: {self.args.model.knowledge_usage}, {self.args.model.use_description}', flush=True)

        if raw_item["text_in"]:
            # print(f'1111',flush=True)
            ###################
            # With text input #
            ###################
            if self.conv_sep in raw_item["text_in"]:
                # print(f'222', flush=True)
                ##################
                # Conversational #
                ##################
                # TODO (commented by Chen): the context part roughly follows the implementation of CoSQL by Tianbao.
                # text_in = "[utt n] || [utt n-1] | [utt n-2] | ..."
                index = raw_item["text_in"].index(self.conv_sep)
                if self.args.model.knowledge_usage == 'concatenate' or self.args.model.knowledge_usage is None:
                    # seq_in  = "[utt n] ; structured knowledge: struct_in ; context: [utt n-1] | [utt n-2] | ..."
                    seq_in = "{} ; structured knowledge: {} ; context: {}".format(raw_item["text_in"][:index],
                                                                                  raw_item["struct_in"],
                                                                                  raw_item["text_in"][index + len(self.conv_sep):])
                elif self.args.model.knowledge_usage == 'separate':
                    # seq_in  = "[utt n] ; context: [utt n-1] | [utt n-2] | ..."
                    seq_in = "{} ; context: {}".format(raw_item["text_in"][:index],
                                                       raw_item["text_in"][index + len(self.conv_sep):])
                else:
                    raise ValueError()
            else:
                ######################
                # Non-conversational #
                ######################
                if self.args.model.knowledge_usage == 'concatenate' or self.args.model.knowledge_usage is None:
                    # seq_in  = "text_in ; structured knowledge: struct_in"
                    # seq_in = "{} ; structured knowledge: {}".format(raw_item["text_in"], raw_item["struct_in"])
                    seq_in = "question: {} ; structured knowledge: {}".format(raw_item["text_in"], raw_item["struct_in"])
                elif self.args.model.knowledge_usage == 'separate':
                    # seq_in  = "text_in"
                    seq_in = raw_item["text_in"]
                else:
                    raise ValueError()
        else:
            ######################
            # Without text input #
            ######################
            if self.args.model.knowledge_usage == 'concatenate':
                # seq_in  = "structured knowledge: struct_in"
                seq_in = "structured knowledge: {}".format(raw_item["struct_in"])
            elif self.args.model.knowledge_usage == 'separate':
                # seq_in  = ""
                seq_in = ""
            else:
                raise ValueError()

        # Concatenate description.
        if self.args.model.use_description and self.args.model.concatenate_description:
            seq_in = "{} ; {}".format(raw_item["description"], seq_in)

        tokenized_question_and_schemas = self.tokenizer(
            seq_in,
            padding="max_length",
            truncation=True,
            max_length=self.training_args.max_length,
            # We found that set it as large as possible can boost the performance significantly
            # , meanwhile, due to the t5 uses a relative position coding, we need to manually
            # assign the max input length into some large numbers, instead of using the "max_model_length"
            # ,which the default is 512, which will hurt the performance a lot.
        )
        seq_out = "answer: {}".format(raw_item['seq_out'])
        tokenized_inferred = self.tokenizer(
            seq_out,
            padding="max_length",
            truncation=True,
            max_length=self.training_args.target_max_length,
            # We set the max_length of "seq_out" during training is the same with the one in inference.
        )

        tokenized_inferred_input_ids = torch.LongTensor(tokenized_inferred.data["input_ids"])
        # Here -100 will let the model not to compute the loss of the padding tokens.
        tokenized_inferred_input_ids[tokenized_inferred_input_ids == self.tokenizer.pad_token_id] = -100

        if self.datatype == 'train':
            item = {
                'input_ids': torch.LongTensor(tokenized_question_and_schemas.data["input_ids"]),
                'attention_mask': torch.LongTensor(tokenized_question_and_schemas.data["attention_mask"]),
                'labels': tokenized_inferred_input_ids,
            }
        else:
            item = {
                'input_ids': torch.LongTensor(tokenized_question_and_schemas.data["input_ids"]),
                'attention_mask': torch.LongTensor(tokenized_question_and_schemas.data["attention_mask"]),
                'labels': tokenized_inferred_input_ids,
                'raw_data': raw_item
            }
         # Add task name.
        if 'task_id' in raw_item:
            item['task_ids'] = raw_item['task_id']

        # Separate description tokenization.
        if self.args.model.use_description and self.args.model.map_description:
            tokenized_description = self.tokenizer(raw_item["description"],
                                                   padding="max_length",
                                                   truncation=True,
                                                   max_length=self.args.dataset.description_max_length,
                                                   )
            item['description_input_ids'] = torch.LongTensor(tokenized_description.data["input_ids"])
            item['description_attention_mask'] = torch.LongTensor(tokenized_description.data["attention_mask"])

        # Separate knowledge tokenization.
        if self.args.model.knowledge_usage == 'separate':
            tokenized_knowledge = self.tokenizer(raw_item["struct_in"],
                                                 padding="max_length",
                                                 truncation=True,
                                                 max_length=self.training_args.max_length,
                                                 )
            item['knowledge_input_ids'] = torch.LongTensor(tokenized_knowledge.data["input_ids"])
            item['knowledge_attention_mask'] = torch.LongTensor(tokenized_knowledge.data["attention_mask"])

        return item

    def __len__(self):
        return len(self.seq2seq_dataset)

def skg_eval(model, tokenizer, eval_dataloader, eval_examples, device, args, epoch, evaluator, num_eval_samples=100, step=0, eval_part=True):
    """
    Change the evaluation implementation of SKG tasks from UnifiedSKG trainer.py and transformers trainer.py
    """
    print(f'Begin evaluating on device {device}', flush=True)
    model = model.to(device)
    model.eval()
    
    #* Evaluation loop to get output.predictions
    preds_host = None
    all_preds = None

    gen_kwargs = {
        'max_length': args.target_max_length,
        'num_beams': args.generation_num_beams 
    }
    # if device == 0:
    #     all_scores = defaultdict(list)
    
    if eval_part:
        stage = "eval_part_epoch{}_step{}".format(epoch, step)
    else:
        stage = "eval_all_epoch{}_step{}".format(epoch, step)

    observed_num_examples = 0
    print_output = True

    for step, batch in enumerate(eval_dataloader):
        batch_raw = batch['raw_data']
        batch = to_device(batch, device)

        observed_batch_size = find_batch_size(batch)
        # print(f'len raw {len(batch_raw)}; observed_btz {observed_batch_size}', flush=True)
        # model_inputs = {k: batch[k] for k in ["input_ids", "attention_mask", "labels"]}
        
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

        model = unwrap_model(model)
        with torch.no_grad():
            # outs = model(**batch)
            generated_tokens = model.generate(
                inputs=batch['input_ids'], 
                attention_mask=batch["attention_mask"],
                **gen_kwargs)

        if gen_kwargs.get("max_length") is not None and generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = _pad_tensors_to_max_len(tokenizer, generated_tokens, gen_kwargs["max_length"])
        
        logits = generated_tokens
        # print(logits, flush=True)

        logits = nested_detach(logits)
        batch_pred = nested_numpify(logits)

        if observed_batch_size is not None:
            observed_num_examples += observed_batch_size

        eval_preds = post_process_function(tokenizer, batch_raw, batch_pred, stage, args, device, print_output)
        print_output = False

        #? Debug--------
        if eval_part:
            # if step >= 5:
            if step >= len(eval_dataloader) // 5:
                break
        #? -------------

    #* Load predictions and calculate scores.
    if eval_part:
        if device == 0:    
            with open(f"{args.output_dir}/{stage}.json", "r") as f:
                res = [json.loads(i) for i in f.readlines()]
            all_preds = [ex['prediction'] for ex in res]
            all_labels = res
            final_scores = compute_metrics(evaluator, all_preds, all_labels, section='dev')
        else:
            final_scores = None
        if device != 'cuda': dist.barrier()
        return final_scores
    else:
        return None

def compute_metrics(evaluator, eval_predictions, eval_labels, section):
    return evaluator.evaluate(eval_predictions, eval_labels, section)

def post_process_function(tokenizer, examples, predictions, stage, args, device, print_output=False):
    # assert isinstance(examples, Dataset)

    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    targets = [ex['seq_out'] for ex in examples]

    if print_output:
        if device == 0 or device == 'cuda':
            for i in range(len(predictions)):
                print(f'device {device};{i}th pred: {predictions[i][8:]}',flush=True)
                print(f'device {device};{i}th tgt: {targets[i]}', flush=True)

    # if device == 2:
    #     for i in range(len(predictions)):
    #         print(f'device {device}; {i}th pred: {predictions[i]}',flush=True)
    #         print(f'device {device};{i}th tgt: {targets[i]}', flush=True)

    # if device == 5:
    #     print(len(predictions), len(targets), flush=True)
    #     for i in range(len(predictions)):
    #         print(f'device {device};{i}th pred: {predictions[i]}',flush=True)
    #         print(f'device {device};{i}th tgt: {targets[i]}', flush=True)
    
    #* Save locally.
    if device == 'cuda' or (type(device)==int and device<=0):
        with open(f"{args.output_dir}/{stage}.json", "a") as f:
            for idx in range(len(predictions)):
                print(json.dumps(dict(**{"prediction": predictions[idx][8:]}, **examples[idx])), file=f) # len("answer: ")=8

    return EvalPrediction(predictions=predictions, items=examples)

def _pad_across_processes(tensor, pad_index=-100):
    """
    Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
    they can safely be gathered.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(_pad_across_processes(t, pad_index=pad_index) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: _pad_across_processes(v, pad_index=pad_index) for k, v in tensor.items()})
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Can't pad the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
        )

    if len(tensor.shape) < 2:
        return tensor
    # Gather all sizes
    size = torch.tensor(tensor.shape, device=tensor.device)[None]
    sizes = size.cpu()

    max_size = max(s[1] for s in sizes)
    # When extracting XLA graphs for compilation, max_size is 0,
    # so use inequality to avoid errors.
    if tensor.shape[1] >= max_size:
        return tensor

    # Then pad to the maximum size
    old_size = tensor.shape
    new_size = list(old_size)
    new_size[1] = max_size
    new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
    new_tensor[:, : old_size[1]] = tensor
    return new_tensor

def _nested_gather(tensors, args=None):
    """
    Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
    concatenating them to `gathered`
    """
    if tensors is None:
        return

    if args.local_rank != -1:
        tensors = distributed_concat(tensors)
    return tensors

def _pad_tensors_to_max_len(tokenizer, tensor, max_length):
    if tokenizer is not None and hasattr(tokenizer, "pad_token_id"):
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
    else:
        raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

    padded_tensor = pad_token_id * torch.ones(
        (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, : tensor.shape[-1]] = tensor
    return padded_tensor

def skg_eval_data_collator(features):
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids", "raw_data") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        elif k == 'raw_data':
            batch[k] = [f['raw_data'] for f in features]

    return batch