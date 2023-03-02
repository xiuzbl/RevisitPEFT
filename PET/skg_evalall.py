import os, sys
import json
import torch
import glob
import argparse
import parse
from skgutils.tool import get_constructor, get_evaluator
from skgutils.configue import Configure
from torch.utils.tensorboard import SummaryWriter

dir_path='/azure/yingxiu/DATA/PET/outputs/0219_train_local_adapter_unified_skg-task_spider_ngpu8_t5-base_normal_tbgrads_lr3e-3_run0/'

def compute_metrics(evaluator, eval_predictions, eval_labels, section):
    return evaluator.evaluate(eval_predictions, eval_labels, section)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skg_cfg', type=str, default=None)
    parser.add_argument('--eval_dir', type=str, default=None)
    parser.add_argument('--eval_type', type=str, default='all')
    parser.add_argument('--tbdir',type=str, default=None)

    args = parser.parse_args()

    dir_path = args.eval_dir
    skgargs = Configure.Get(args.skg_cfg)
    evaluator = get_evaluator(skgargs.evaluate.tool)(skgargs)
    score_path = os.path.join(args.eval_dir, args.eval_type+'_scores.txt')
    valid_writer = SummaryWriter(os.path.join(args.tbdir, 'eval'))

    #* Get prediction files from the folder
    # files=glob.glob(os.path.join(dir_path, 'eval_'+args.eval_type+'_*.json'))
    # stage_list = []
    # for f in files:
    #     # print(f)
    #     res = parse.parse(dir_path+'eval_'+args.eval_type+'_epoch{}_step{}.json', f)
    #     eval_stage = [int(k) for k in res.fixed]
    #     stage_list.append(eval_stage)
    # stage_list.sort(key=lambda x: (x[0], x[1]))
    # print(f'There are {len(stage_list)} predictions.')

    #* Evaluate the predictions for stages stored in the folder
    # if os.path.exists(score_path):
    #     os.remove(score_path)

    # with open(score_path, 'a') as fw:
    #     for epoch, step in stage_list: 
    #         filename = dir_path+'eval_'+args.eval_type+'_epoch{}_step{}.json'.format(epoch, step)
    #         # print(filename)
    #         print(f'Begin evaluating for file {filename}', flush=True)
    #         with open(filename, 'r') as fr:
    #             data = [json.loads(i) for i in fr.readlines()]
    #         predictions = [ex['prediction'] for ex in data]
    #         labels = data
    #         final_scores = compute_metrics(evaluator, predictions, labels, section='dev')
    #         final_scores['step'] = step
    #         final_scores['epoch'] = epoch

    #         print(json.dumps(final_scores),file=fw)
    # print(f'Finish collecting scores to {score_path}')

    #TODO write scores to tensorboard
    with open(score_path, 'r') as fr:
        res = [json.loads(i) for i in fr.readlines()]
        for row in res:
            score = row['avr']
            step = row['step']
            valid_writer.add_scalar('eval/avr', score, step)
            valid_writer.add_scalar('eval/score', score, step)
    print(f'Finish writing scores to {args.tbdir}')
    print('Congrats!!!')

