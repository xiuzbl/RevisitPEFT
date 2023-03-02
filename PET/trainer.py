from myutils import *
from arguments import *
import json, os, sys, logging, torch, math, time
import numpy as np
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    set_seed
)
import transformers, datasets
import deepspeed

print(f'TORCH CUDA:', torch.cuda.is_available(), flush=True)
# os.environ["XLA_FLAGS"] = '--xla_gpu_force_compilation_parallelism=1'


def main():
    #* Setup logging
    logging.basicConfig(
        format="[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] >> %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S")
    logger = logging.getLogger(__name__)
    logging.Formatter.converter = time.localtime
    logger.error("localtime")
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data Arguments: {data_args}")
    logger.info(f"Model Arguments: {model_args}")
    logger.info(f'My Arguments: {myargs}')

    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        mirror='tuna',
    )


    #* Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer, revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        model_max_length=data_args.max_source_length
    )

    # model = AutoModelForSeq2SeqLM.from_pretrained(
    model = MyModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,)

    special_tokens = ["<last_turn>", "<user>", "<agent>", "<grounding>","<title>","</title>"]
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    model = model.cuda()

    logger.info("Begin Data Preprocessing ...")

    model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                     model=model,
                                                     model_parameters=params)

    #* Load checkpoint
    _, client_sd = model_engine.load_checkpoint(args.load_dir, args.ckpt_id)
    step = client_sd['step']

    #advance data loader to ckpt step
    dataloader_to_step(data_loader, step + 1)

    for step, batch in enumerate(data_loader):

        #forward() method
        loss = model_engine(batch)

        #runs backpropagation
        model_engine.backward(loss)

        #weight update
        model_engine.step()

        #save checkpoint
        if step % args.save_interval:
            client_sd['step'] = step
            ckpt_id = loss.item()
            model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd = client_sd)
