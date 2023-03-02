# a100=true
a100=false
# port=1123
# port=10000
port=1234
echo PORT: $port
now="$(date +'%m%d')"
echo DATE: $now
# now=0128

runid=0
# runid=single0
#gpuid=0,1,2,3
gpuid=0,1,2,3,4,5,6,7
# gpuid=8,9,10,11,12,13,14,15
# gpuid=0
# gpuid=4,5,6,7
ngpu=8
echo GPUID: $gpuid
echo Num_gpus: $ngpu

# add_peft=true
add_peft=
peft=adapter
adapter_reduction=6

# debug=true
debug=false

epochs=1
grad_accu=4
# lr=1e-5
lr=5e-5
# warmup_ratio=0.1
warmup_ratio=0.0

# model_name=t5-base
# model_name=t5-3b
model_name=flan-t5-xl
# model_name=t5-11b
# do_train=true
do_train=
# do_eval=true
do_eval=
# task_type=t0
task_type=unified_skg

train_btz=4
eval_btz=1
exp=${now}_

# max_input=512
# max_input=450
max_input=256
# max_input=128
# max_input=1024
# max_target=256
# max_target=60
max_target=128
# padding=true
num_beams=1
padding=

if [[ "$task_type" = t0 ]]
then
    dataset=super_glue
    dataset_config=rte
    extra_cmd=""
    # extra_cmd="--use_template"
    template_name="does this imply"
    # template_name="based_on_that"
    echo Dataset: $dataset $dataset_config
    echo Use template: $extra_cmd
    echo Template: $template_name
else
    echo Unified SKG tasks
    extra_cmd=""
    dataset=spider
    # dataset_config=spider
    # dataset=sql2text
    # dataset=sqa
    # dataset=wikisql
    # dataset=wikitq
    dataset_config=$dataset
    # dataset_config=sql2text
    template_name="none"
fi

#! Debug-------------------------
# if [[ -z "$debug" ]]
if $debug
then
    max_train_steps=0
    # max_train_steps=5000
    echo DEBUGGING... 
    # echo Max train steps: $max_train_steps
    extra_cmd+=" --debug"
    exp+="debug_zero_"
    eval_step=50
    num_shots=10
    # exp=debug_wandb_
else
    max_train_steps=20000
    # echo FORMAL TRAINING
    # echo Max train steps: $max_train_steps
    exp+="zero_"
    num_shots=1000000
    eval_step=100
fi


#! PEFT Module----------------------
if [[ -z $add_peft ]]
then
    # echo Finetune all parameters... NO PEFT
    echo No Finetuning
    method=finetune
else
    echo Use PEFT: $peft
    method=$peft
    # method=finetune_$peft
fi

echo Method: $method
model_path=/azure/yingxiu/DATA/${model_name}
# model_path=/azure/yingxiu/Yingxiu_Intern/UnifiedSKG/output/T5_base_finetune_spider
# model_path=/azure/yingxiu/Yingxiu_Intern/UnifiedSKG/output/T5_base_finetune_spider/checkpoint-7500
# model_path=$model_name
# model_path="t5-3b"
# model_path=/azure/yingxiu/DATA/t5-3b
echo MODEL: $model_path

#! Experiment
# exp=ds_data-${dataset}_${dataset_config}_ngpu-${ngpu}_model-${model_name}_run0
# exp=${now}_${method}_${dataset}_${dataset_config}_ngpu${ngpu}_model_${model_name}_run0
exp+="${task_type}-task_${dataset}_ngpu${ngpu}_model_${model_name}_run$runid"
# exp=0128_finetune_unified_skg-task_spider_ngpu8_model_t5-base_run0
# exp=${peft}_data${dataset_config}_ngpu${ngpu}_model_${model_name}_run0
echo EXP: $exp

#? Tasks configures-------------------------------------
config=./configs/${method}.json+./configs/rte.json
# skg_cfg=META_TUNING/spider.cfg
skg_cfg=Salesforce/T5_base_finetune_spider_with_cell_value.cfg
# skg_cfg=Salesforce/T5_base_finetune_${dataset}.cfg
echo SKG-TASK: $skg_cfg
# ds_config=dsconfigs/t53b_peft.json
# ds_config=dsconfigs/t53b_icl.json
ds_config=dsconfigs/t53b_wikisql_icl.json
#?--------------------------------------------------------

#* Output and Logs -----------------------------------------------------------
pet=/azure/yingxiu/DATA/PET
output=${pet}/outputs/$exp
log=${pet}/logs/$exp.log
tbdir=${pet}/tb_logs/$exp
cache_dir=${pet}/cache_dir/$exp
mkdir -p ${pet}/cache_dir $cache_dir
mkdir -p ${pet}/outputs $output ${pet}/logs ${pet}/tb_logs/$exp

# TODO: Training ..............................................................
# py=/azure/yingxiu/ENVS/pet/bin/python
# ds=/azure/yingxiu/ENVS/pet/bin/deepspeed
# ds=/azure/yingxiu/ENVS/tzero/bin/deepspeed
ds=/opt/conda/envs/pet/bin/deepspeed

export CUDA_VISIBLE_DEVICES=$gpuid
# $ds --include="worker-0:${gpuid}" --master_port=$port \
# $py -m torch.distributed.launch --nproc_per_node=$ngpu --master_port=$port \
#     icl_run.py \
$ds --num_gpus=$ngpu \
    zero_dsrun.py \
    --deepspeed_config $ds_config \
    --deepspeed \
    --dataset_name=$dataset \
    --dataset_config_name=$dataset_config \
    --config_files=$config \
    --model_name_or_path=$model_path \
    --per_device_train_batch_size=$train_btz \
    --per_device_eval_batch_size=$eval_btz \
    --num_train_epochs=$epochs \
    --seed 42 \
    --output_dir=$output \
    --tbdir=$tbdir \
    --cache_dir=$cache_dir \
    --do_train=$do_train \
    --do_eval=$do_eval \
    --template_name="$template_name" \
    --max_train_steps=$max_train_steps \
    --gradient_accumulation_steps=$grad_accu \
    --eval_step=$eval_step \
    --max_length=$max_input \
    --target_max_length=$max_target \
    --pad_to_max_length=$padding \
    --add_peft=$add_peft \
    --master_port=$port \
    --skg_task \
    --skg_cfg=$skg_cfg \
    --num_shots=$num_shots \
    --generation_num_beams=$num_beams \
    -k exp_name=$exp lr=$lr warmup_ratio=$warmup_ratio exp_dir=$output adapter_reduction_factor=$adapter_reduction lora_rank=$lora_rank \
    ${extra_cmd} \
    >$log 2>&1
