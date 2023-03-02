import argparse
import logging
import os
import random
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import csv
import math
import datasets
import torch
from torch.utils.data import DataLoader
import evaluate

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    default_data_collator,
    DataCollatorForSeq2Seq,
    AdamW,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import PaddingStrategy
from promptsource.templates import DatasetTemplates

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
            Note that it's very NOT recommended to use fp16 to do any time of inference with T0 as the predictions will vastly differ from the predictions using fp32.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [
                {
                    k: v[i]
                    for k, v in feature.items()
                    if k != "targets"
                }
                for i in range(num_choices)
            ]
            for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Pad the labels because it's not padded automatically
        max_label_length = max([len(elem["labels"]) for elem in flattened_features])
        batch["labels"] = [
            l + [self.tokenizer.pad_token_id]*(max_label_length - len(l))
            for l in [elem["labels"] for elem in flattened_features]
        ]
        batch["labels_attention_mask"] = [
            m + [0]*(max_label_length - len(m))
            for m in [elem["labels_attention_mask"] for elem in flattened_features]
        ]

        # Convert to tensors
        batch = {
            k: torch.tensor(v)
            for k, v in batch.items()
        }

        batch["targets"] = torch.tensor([f.pop("targets") for f in features])
        return batch

def get_input_target(args, sample):
    dataset_name = args.dataset_name
    dataset_config_name = args.dataset_config_name

    if dataset_name == 'super_glue':
        if dataset_config_name == 'rte':
            label_dict = DATASET_LABELS[dataset_name][dataset_config_name]
            # input = sample['premise'] + ' ' + sample['hypothesis']
            input = 'premise:' + sample['premise'] + ' ' + 'hypothesis:' + sample['hypothesis']
            target = label_dict[sample['label']]
            label_list = list(label_dict.values())
    elif dataset_name == 'xsum':
        input = sample['document']
        target = sample['summary']
        label_list = None

    return input, target, label_list

def t0_preprocess_train(examples, tokenizer, template, column_names, args):
    bs = len(examples[column_names[0]])
    padding = "max_length" if args.pad_to_max_length else False

    input_texts = []
    target_texts = []
    for i in range(bs):
        # example (ex)
        ex = {k: examples[k][i] for k in column_names}
        if args.use_template:
            input, target = template.apply(ex)
            label_list = template.get_answer_choices_list(ex)
        else:
            input, target, label_list = get_input_target(args, ex)
        assert target in label_list
        #print(f'target {target}\n {ex_answer_choices}', flush=True)
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

def t0_preprocess_eval(examples, tokenizer, template, column_names, args):
    bs = len(examples[column_names[0]])
    padding = "max_length" if args.pad_to_max_length else False

    input_texts = []
    target_texts = []
    answer_choices_texts = []
    for i in range(bs):
        ex = {
            k: examples[k][i]
            for k in column_names
        }
        if args.use_template:
            input, target = template.apply(ex)
            label_list = template.get_answer_choices_list(ex)
        else:
            input, target, label_list = get_input_target(args, ex)

        # print(f'target {target}' ,flush=True)
        # print(f'label_list {label_list}', flush=True)
        try:
            assert target in label_list
        except:
            print(f'target "{target}" not in label list: {label_list}', flush=True)
            continue
        
        input_texts.append(input)
        target_texts.append(target)
        answer_choices_texts.append(label_list)

    bs = len(input_texts)

    tokenized_inputs = tokenizer(
        input_texts,
        padding=padding,
        max_length=args.max_length,
        truncation=True,
        add_special_tokens=True,
    )
    tokenized_targets = [
        tokenizer(
            ans_choi,
            padding=True,
            max_length=args.target_max_length,
            truncation=True,
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

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

def t0_eval(model, ddloader, device):
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



def process_nlu_eval():
    return

def process_xsum_eval():
    return

EVAL_PROCESS_DICT = {
    'nlu': process_nlu_eval,
    'xsum': process_xsum_eval,
}


DATASET_LABELS = {
    'super_glue':{
        'rte':{
            0:'YES',
            1:'NO',
        }
    }
}