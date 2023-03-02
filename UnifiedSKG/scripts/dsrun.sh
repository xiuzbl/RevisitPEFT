gpuid=0,1,2,3,4,5,6,7
# gpuid=8,9,10,11,12,13,14,15
# gpuid=4,5,6,7
ngpu=8
now="$(date +'%m%d')"
echo DATE $now
echo GPUID: $gpuid
echo NUM GPUs: $ngpu

train_btz=1
echo train_batch_size: $train_btz
input_max_length=512
target_max_length=256
beam_size=1
# input_max_length=1024
# target_max_length=128
# beam_size=4
echo Input_max_length: $input_max_length
echo Target_max_length: $target_max_length
adapter_reduction=6
eval_steps=100

run_name=ds_${now}_T5-base_adapter_spider_reduction${adapter_reduction}_run1
# run_name=ds_T5-3b_finetune_spider_run1
# run_name=ds_T5-3b_adapter_spider_reduction${adapter_reduction}_run01
# run_name=T5-base_prefix_spider
# run_name=ds_T5-3b_finetune_sqa_run0
echo RUN EXP: $run_name

# cfg=Salesforce/T5_base_finetune_wikitq.cfg
cfg=Salesforce/T5_base_adapter_spider_with_cell_value.cfg
# cfg=Salesforce/T5_3b_finetune_spider_with_cell_value.cfg
# cfg=Salesforce/T5_3b_adapter_spider_with_cell_value.cfg
# cfg=Salesforce/T5_3b_finetune_sqa.cfg
# cfg=Salesforce/T5_base_prefix_spider_with_cell_value.cfg
echo config file: $cfg

# export WANDB_API_KEY="9bb23cf3c9acc6172c9d16d4045b0262f78376c6"
# export WANDB_PROJECT=uniskg
# export WANDB_ENTITY=niesheng
log=logs/$run_name.log
output=output/$run_name
tblog=tb_logs/$run_name
mkdir -p $output tb_logs $tblog

# py=/home/v-yingxzhao/.conda/envs/py3.7pytorch1.8new/bin/python
ds=/azure/yingxiu/ENVS/uniskg/bin/deepspeed
echo which deepspeed: $ds

# CUDA_VISIBLE_DEVICES=$gpuid \
# $ds \
#     --num_gpus=$ngpu \
# $ds --include="worker-0:8,9,10,11,12,13,14,15" --master_port 60000 \
$ds --include="worker-0:0,1,2,3,4,5,6,7" --master_port 22000 \
    train.py \
    --deepspeed deepspeed/ds_config_zero2.json --seed 42 \
    --cfg=$cfg \
    --run_name=$run_name \
    --logging_strategy steps --logging_first_step true --logging_steps 4 \
    --evaluation_strategy steps --eval_steps=$eval_steps --metric_for_best_model avr --greater_is_better true \
    --save_strategy steps --save_steps 500 --save_total_limit 5 --load_best_model_at_end \
    --gradient_accumulation_steps 16 --num_train_epochs 50 --adafactor false --learning_rate 5e-5 \
    --do_train --do_eval --do_predict \
    --predict_with_generate \
    --output_dir output/$run_name --overwrite_output_dir \
    --per_device_train_batch_size=$train_btz --per_device_eval_batch_size 1 \
    --generation_num_beams=$beam_size \
    --generation_max_length=$target_max_length \
    --input_max_length=$input_max_length \
    --ddp_find_unused_parameters true \
    --report_to="tensorboard" \
    --logging_dir=$tblog \
    >$log 2>&1 
