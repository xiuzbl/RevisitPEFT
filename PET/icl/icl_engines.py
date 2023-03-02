from t0dataset import DataCollatorForMultipleChoice, get_input_target, t0_preprocess_train, t0_preprocess_eval, t0_eval

from torch.utils.data import DataLoader, Dataset
from transformers import  DataCollatorForSeq2Seq
import json
import torch
import sys
sys.path.append('../')
from src.data import FinetuneDatasetWithTemplate, create_collate_fn
# from src.data import FinetuneDatasetWithTemplate, create_collate_fn
from tqdm import tqdm


def get_icl_engine(ticl_config, dataset_reader, tokenizer):
    return ICLEngine(ticl_config, dataset_reader, tokenizer)


class ICLEngine(object):
    def __init__(self, ticl_config, dataset_reader, tokenizer, model):
        self.config = ticl_config
        self.dataset_reader = dataset_reader
        self.tokenizer = tokenizer
        self.model = model
        self.train_loader, self.eval_loader, self.boilerplates = self._prepare_data()

    def run(self, model):
        outputs = []
        for batch in tqdm(self.eval_loader):
            outputs.append(self._predict(model, batch))
        self._evaluate(outputs)

    def _prepare_data(self):
        # if self.config.num_shot > 0:
        raw_datasets = get_dataset(args)
        if args.debug:
            if torch.distributed.get_rank() == 0: logger.info(f'Debugging......')
            raw_train_dataset, raw_eval_dataset = raw_datasets['validation'], raw_datasets['validation']
        else:
            raw_train_dataset, raw_eval_dataset = raw_datasets['train'], raw_datasets['validation']
        if torch.distributed.get_rank() == 0: logger.info(f'Number of Raw Train-set {raw_train_dataset}; Raw Eval-set {raw_eval_dataset}')

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
 
        train_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model, label_pad_token_id=-100, pad_to_multiple_of=None)
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=train_collator,  batch_size=self.config.per_device_train_batch_size)

        if self.config.pad_to_max_length or self.config.skg_task:
            # If padding was already done ot max length, we use the default data collator that will just convert everything to tensors.
            eval_collator = default_data_collator
        else: # T0 task data collator
            # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
            # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
            # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            eval_collator = DataCollatorForMultipleChoice(
                tokenizer, pad_to_multiple_of=None
            )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=eval_collator, batch_size=self.config.per_device_eval_batch_size)
 
        #     self.train_dataset = FinetuneDatasetWithTemplate(
        #         self.dataset_reader.read_orig_dataset("train"),
        #         self.dataset_reader.get_train_template(),
        #         self.tokenizer,
        #         add_special_tokens=False,
        #     )
        #     train_loader = torch.utils.data.DataLoader(
        #         self.train_dataset,
        #         batch_size=self.config.num_shot,
        #         shuffle=True,
        #         collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False),
        #         num_workers=min([self.config.num_shot, self.config.num_workers]),
        #         drop_last=True,
        #     )
        # else:
        #     train_loader = None

        # self.eval_dataset = FinetuneDatasetWithTemplate(
        #     self.dataset_reader.read_orig_dataset("validation"),
        #     self.dataset_reader.get_eval_template(),
        #     self.tokenizer,
        #     add_special_tokens=True,
        # )
        # eval_loader = torch.utils.data.DataLoader(
        #     self.eval_dataset,
        #     batch_size=self.config.eval_batch_size,
        #     shuffle=False,
        #     collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False),
        #     num_workers=min([self.config.eval_batch_size, self.config.num_workers]),
        # )

        def get_tokens(text):
            # force convert to long in case of empty list
            return self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(torch.long)

        boilerplates = {
            "context_start": get_tokens(self.config.context_start),
            "context_input_target_separator": get_tokens(self.config.context_input_target_separator),
            "context_example_separator": get_tokens(self.config.context_example_separator),
            "context_end": get_tokens(self.config.context_end),
        }
        print("Boilerplates")
        print(boilerplates)

        return train_dataloader, eval_dataloader, boilerplates

    def _evaluate(self, outputs):
        accumulated = {key: [] for key in outputs[0].keys()}
        for batch_output in outputs:
            for key, value in batch_output.items():
                accumulated[key].extend(value)
        metrics = self.dataset_reader.compute_metric(accumulated)

        result_str = json.dumps(metrics) + "\n"
        with open(self.config.dev_score_file, "a+") as f:
            f.write(result_str)
        print("\n" + result_str)
        print(f"Results saved to {self.config.dev_score_file}")

    def _get_context(self):
        if self.train_loader is None:
            context = torch.zeros((1, 0), dtype=torch.long)

        else:
            context_batch = next(iter(self.train_loader))

            if self.config.icl_modeling == "channel":
                context_batch["input_ids"], context_batch["target_ids"] = (
                    context_batch["target_ids"],
                    context_batch["input_ids"],
                )
            context_array = torch.cat(
                [
                    context_batch["input_ids"],
                    self.boilerplates["context_input_target_separator"].repeat(self.config.num_shot, 1),
                    context_batch["target_ids"],
                    self.boilerplates["context_example_separator"].repeat(self.config.num_shot, 1),
                ],
                dim=1,
            )

            if self.config.icl_method == "concat":
                context = context_array.flatten()[None, :]
            elif self.config.icl_method == "ensemble":
                context = context_array
            context = self._left_align_tensor(context)[:, : self.config.max_context_len]

            context = torch.cat(
                [
                    self.boilerplates["context_start"].repeat(context.size(0), 1),
                    context,
                    self.boilerplates["context_end"].repeat(context.size(0), 1),
                ],
                dim=1,
            )

        return context

    def _predict(self, model, batch):
        context = self._get_context()  # (1 or num_shot, context_len)
        input_ids = batch["input_ids"]  # (bs, seq_len)
        answer_choices_ids = batch["answer_choices_ids"]  # (bs, num_choices, seq_len)

        bs, num_choices = batch["answer_choices_ids"].size()[:2]
        num_context = context.size(0)

        if self.config.icl_modeling == "channel":
            inputs = torch.cat(
                [
                    context[None, :, None, :].expand(bs, num_context, num_choices, -1),
                    answer_choices_ids[:, None, :, :].expand(bs, num_context, num_choices, -1),
                ],
                dim=3,
            )
            outputs = torch.cat(
                [
                    self.boilerplates["context_input_target_separator"][None, None, :, :].expand(
                        bs, num_context, num_choices, -1
                    ),
                    input_ids[:, None, None, :].expand(bs, num_context, num_choices, -1),
                ],
                dim=3,
            )
        else:
            inputs = torch.cat(
                [
                    context[None, :, None, :].expand(bs, num_context, 1, -1),
                    input_ids[:, None, None, :].expand(bs, num_context, 1, -1),
                ],
                dim=3,
            )
            outputs = torch.cat(
                [
                    self.boilerplates["context_input_target_separator"][None, None, :, :].expand(
                        bs, num_context, num_choices, -1
                    ),
                    answer_choices_ids[:, None, :, :].expand(bs, num_context, num_choices, -1),
                ],
                dim=3,
            )

            if self.config.icl_modeling == "calibration":
                inputs_null = context[None, :, None, :].expand(bs, num_context, 1, -1)
                batch["input_ids_null"] = inputs_null.flatten(start_dim=0, end_dim=-3)

        batch["input_ids"] = self._left_align_tensor(inputs.flatten(start_dim=0, end_dim=-3))
        batch["answer_choices_ids"] = outputs.flatten(start_dim=0, end_dim=-3)

        choices_scores = model(batch)
        # _, pred = choices_scores.view(bs, num_context, num_choices).min(dim=2)
        # acc = pred == batch["labels"]
        # print(acc)
        # self._temp.append(acc)
        choices_scores = choices_scores.view(bs, num_context, num_choices)

        if self.config.icl_modeling == "calibration":
            batch["input_ids"] = batch["input_ids_null"]
            null_scores = model(batch)
            choices_scores = choices_scores - null_scores.view(bs, num_context, num_choices)

        if self.config.icl_method == "ensemble":
            choices_votes = (choices_scores <= choices_scores.min(dim=-1, keepdim=True)[0]).float().mean(dim=1)
            _, prediction = choices_votes.max(dim=1)
        else:
            _, prediction = choices_scores.squeeze(dim=1).min(dim=-1)

        return {
            "prediction": prediction.tolist(),
            "label": batch["labels"].tolist(),
            "idx": batch["idx"],
        }

    def _left_align_tensor(self, tensor):
        tensor_mask = tensor != self.tokenizer.pad_token_id
        tensor_nonpad = tensor_mask.nonzero(as_tuple=True)
        output = torch.arange(tensor.size(-1)).expand_as(tensor) < (tensor_mask).sum(dim=-1, keepdim=True)
        output = output.to(tensor.dtype)
        output_nonzero = output.nonzero(as_tuple=True)
        output.fill_(self.tokenizer.pad_token_id)
        output[output_nonzero] = tensor[tensor_nonpad]
        return output
    
    def compute_metric(self, accumulated):
        data = []
        idxs = accumulated["idx"]
        predictions = accumulated["prediction"]
        for idx, prediction in zip(idxs, predictions):
            data.append({"ID": idx, "Label": self.answer_choices[prediction]}) #todo
        result_df = pd.DataFrame(data=data, columns=["ID", "Label"]).astype({"ID": int, "Label": str})
        result_df.to_csv(self.config.dev_pred_file, index=False)
        matching = [a == b for a, b in zip(accumulated["prediction"], accumulated["label"])]
        accuracy = sum(matching) / len(matching)
        return {"accuracy": accuracy}
