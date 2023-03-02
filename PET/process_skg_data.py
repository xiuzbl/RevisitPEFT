from skgutils.configue import Configure
from skgutils.tool import get_constructor, get_evaluator
from datasets import load_dataset
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skg_cfg', type=str, required=True)
    args = parser.parse_args()
    #? SKG dataset process 
    skgargs = Configure.Get(args.skg_cfg)

    meta_tuning_data = {}
    for task, arg_path in skgargs.arg_paths:
        task_args = Configure.Get(arg_path)
        task_raw_datasets_split = load_dataset(path=task_args.dataset.loader_path,
                                               cache_dir=task_args.dataset.data_store_path)
        data_cache_root = os.path.join(task_args.dataset.data_store_path, 'cache')
        os.makedirs(data_cache_root, exist_ok=True)
        task_seq2seq_dataset_split = get_constructor(task_args.seq2seq.constructor)(task_args).to_seq2seq(task_raw_datasets_split, data_cache_root)

        meta_tuning_data[arg_path] = task_seq2seq_dataset_split

    seq2seq_dataset_split = get_constructor(skgargs.seq2seq.constructor)(skgargs).to_seq2seq(meta_tuning_data)

    print(f'Finish processing data!')