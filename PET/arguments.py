import argparse
from transformers import SchedulerType
from myutils.util import ParseKwargs

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")

    # parser.add_argument('--peft_module', type=str, default='adapter')
    parser.add_argument('--reset_trainable_params', type=bool, default=False,)
    parser.add_argument('--tb_grads', type=bool, default=False,)
    parser.add_argument('--early_stop_steps', type=int, default=-1)
    parser.add_argument('--recover_weights', type=bool, default=False, help='Whether recover the pretrained weights during training.')
    parser.add_argument('--close_fnn_epochs', type=int, default=-1)
    parser.add_argument('--other_params_list', nargs='+', default=None)
    parser.add_argument('--generation_num_beams', type=int, default=1)
    parser.add_argument("--skg_cfg", type=str, default=None, help='Configure file for SKG tasks.')
    parser.add_argument("--skg_task", action='store_true', help='Whether use structured knolwedge grounding task (SKG) in UnifiedSKG paper.')
    parser.add_argument('--use_template', action='store_true', help='Whether use P3 prompts. ')
    parser.add_argument("--wandb_log", type=str, default=None)
    parser.add_argument('--master_port', type=str, default='37589')
    parser.add_argument('--add_peft', type=bool, default=False)
    parser.add_argument('--use_a100', action='store_true')
    parser.add_argument("-c", "--config_files", required=True)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    parser.add_argument('--cache_dir', type=str, default='./cache_dir')
    parser.add_argument('--tbdir', type=str, help='Store tensorboard logs.')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Clip gradients.')
    parser.add_argument(
        "-ns",
        "--num_shots",
        type=int,
        default=None,
        help="Number of training examples for few-shot learning. Default is None, which uses the entire train set.",
    )
    
    parser.add_argument("-tl", "--target_max_length", type=int, default=256,
        help="Target max length. Sequences longer than this will be truncated."
    )
    parser.add_argument("-il", "--max_length", type=int, default=256,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length", default=False, type=bool,
        help="Default=False; If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument("--debug", action='store_true', default=False,  help="Activate debug mode and run training only with a subset of data.")
    parser.add_argument('--debug_infer', action='store_true', default=False, help='Test inference for a single sample.')
    parser.add_argument(
        "-t",
        "--template_name",
        type=str,
        default=None,
        required=True,
        help="The template/prompt name in `promptsource`.",
    )
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=False)
    parser.add_argument('--do_predict', type=bool, default=False)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument("--train_file", type=str, default=None, help="A csv or a json file containing the training data.")
    parser.add_argument("--validation_file", type=str, default=None, help="A csv or a json file containing the validation data.")
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--path-to-model",
        type=str,
        help="Path to fine-tuned model or model identifier from huggingface.co/models.",
        default=None,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    # parser.add_argument(
    #     "--learning_rate",
    #     type=float,
    #     default=5e-5,
    #     help="Initial learning rate (after the potential warmup period) to use.",
    # )
    # parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=100,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="constant_with_warmup",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )

    parser.add_argument("--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files.")
    parser.add_argument("--not_tie_wre", action="store_true", help="tie the last layer and embedding or not.")
    parser.add_argument("--random_ltd", action="store_true", help="enable random-ltd or not.")
    parser.add_argument("--eval_step", type=int, default=10, help="eval step.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--data_folder", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args
