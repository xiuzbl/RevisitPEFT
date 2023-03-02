# gpuid=0,1,2,3,4,5,6,7,8,9,10,11
#gpuid=12,13,14,15
# ngpu=12
a100=false
# a100=true
# date=0118
# now="$(date +'%m%d')"
now=0128

# gpuid=0,1,2,3
# gpuid=8,9,10,11
# ngpu=4
# gpuid=8
# ngpu=1
gpuid=0,1,2,3,4,5,6,7
ngpu=8

# gpuid=$1
# runid=$2
runid=0
echo GPUID: $gpuid
echo Num GPUs: $ngpu

# exp=${date}_debug_eval_finetune
# exp=${date}_debug_eval_finetune_notemplate_run$runid
#exp=ds_adapter_t5base_test0
# exp=ds_adapter_t511b_test0
#exp=ds_adapter_t5base_test1
#exp=ds_adapter_t511b_test1
#exp=test_t511b
#exp=test_t5base
# exp=test_lora0
#exp=test_adapter03
# exp=test_adapter02
# exp=test_bitfit0
# exp=occupy
# add_peft=true
add_peft=
peft=adapter
# peft=lora
adapter_reduction=6
# lora_rank=4

#! Debug or not
debug=true
# debug=false

epochs=60
grad_accu=4
lr=1e-5
warmup_ratio=0.1
do_train=true
do_eval=
# task_type=t0
task_type=unified_skg

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
    dataset_config=spider
    template_name="none"
fi

# dataset=yelp_review_full
# dataset_config=
train_btz=1
eval_btz=2
#! Debug-------------------------
if $debug
then
    # max_train_steps=500
    max_train_steps=5000
    echo DEBUGGING... 
    echo Max train steps: $max_train_steps
    extra_cmd+=" --debug"
    exp=debug_
    eval_step=10
    # exp=debug_wandb_
else
    max_train_steps=20000
    echo FORMAL TRAINING
    echo Max train steps: $max_train_steps
    exp=train_
    eval_step=100
fi
# max_input=1024 
max_input=512
max_target=256
# padding=true
padding=

#! Experiment----------------------
if $a100
then
    echo Use A100~
    ds=deepspeed
    model_name=t5-11b
    exp+='a100_'
else
    echo Use V100~
    ds=/azure/yingxiu/ENVS/pet/bin/deepspeed
    # ds=deepspeed
    model_name=t5-base
    # model_name=t5-3b
    exp+="v100_"
fi

#! PEFT Module----------------------
if [[ -z $add_peft ]]
then
    echo Finetune all parameters... NO PEFT
    method=finetune
else
    echo Use PEFT: $peft
    method=$peft
fi

model_path=/azure/yingxiu/DATA/${model_name}
# model_path=$model_name
# model_path="t5-3b"
# model_path=/azure/yingxiu/DATA/t5-3b
echo MODEL: $model_path
# exp=ds_data-${dataset}_${dataset_config}_ngpu-${ngpu}_model-${model_name}_run0
# exp=${now}_${method}_${dataset}_${dataset_config}_ngpu${ngpu}_model_${model_name}_run0
exp=${now}_${method}_${task_type}-task_${dataset}_ngpu${ngpu}_model_${model_name}_run0
# exp+="${now}_${method}_${dataset}_${dataset_config}_ngpu${ngpu}_model_${model_name}_reduction${adapter_reduction}_run${runid}"
# exp+="${now}_${method}_${dataset}_${dataset_config}_ngpu${ngpu}_model_${model_name}_lorarank${lora_rank}_run${runid}"
#model_name=t5-base
# model_name=t5-3b
# exp=date${now}_${peft}_${dataset}_${dataset_config}_ngpu${ngpu}_model_${model_name}_run1
echo EXP: $exp
config=./configs/${method}.json+./configs/rte.json
ds_config=dsconfigs/ds_peft_config.json

#* SKG tasks configures
# skg_cfg=META_TUNING/spider.cfg
skg_cfg=Salesforce/T5_base_finetune_spider_with_cell_value.cfg
echo skg task $skg_cfg

#* Output and Logs -----------------------------------------------------------
pet=/azure/yingxiu/DATA/PET
output=${pet}/outputs/$exp
log=${pet}/logs/$exp.log
tbdir=${pet}/tb_logs/$exp
wandb=${pet}/wandblogs/$exp
cache_dir=${pet}/cache_dir/$exp
mkdir -p ${pet}/cache_dir $cache_dir ${pet}/wandblogs ${wandb_log}
mkdir -p $pet ${pet}/outputs $output ${pet}/logs ${pet}/tb_logs $tbdir

# TODO: Training ..............................................................
# py=/azure/yingxiu/ENVS/pet/bin/python
#ds=/mnt/users/yingxiu/pet/bin/deepspeed
# ds=deepspeed
# ds=/azure/yingxiu/ENVS/tzero/bin/deepspeed
# $py -m torch.distributed.launch --nproc_per_node=$ngpu \
# export CUDA_VISIBLE_DEVICES=$gpuid
# $ds \
    # --num_gpus=$ngpu \
#$ds --include="worker-0:8,9,10,11,12,13,14,15" --master_port 60000 \
$ds --include="worker-0:0,1,2,3,4,5,6,7"  \
    ds_run.py \
    --dataset_name $dataset \
    --dataset_config_name $dataset_config \
    --config_files=$config \
    --model_name_or_path $model_path \
    --per_device_train_batch_size=$train_btz \
    --per_device_eval_batch_size=$eval_btz \
    --num_train_epochs=$epochs \
    --deepspeed_config $ds_config \
    --deepspeed \
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
    --wandb_log=$wandb \
    --skg_task \
    --skg_cfg=$skg_cfg \
    -k exp_name=$exp lr=$lr warmup_ratio=$warmup_ratio exp_dir=$output adapter_reduction_factor=$adapter_reduction lora_rank=$lora_rank \
    ${extra_cmd} \
    >$log 2>&1
