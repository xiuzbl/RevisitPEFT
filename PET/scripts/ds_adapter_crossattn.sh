# a100=false
singularity=false
# singularity=true
# a100=true
# date=0118
# port=1120
port=60000
echo PORT: $port
now="$(date +'%m%d')"
echo DATE: $now

runid=0
# ngpu=4
# gpuid=8
# ngpu=1
# gpuid=8,9,10,11,12,13,14,15
gpuid=0,1,2,3,4,5,6,7
ngpu=8
echo GPUID: $gpuid
echo Num GPUs: $ngpu

add_peft=true
# add_peft=
peft=adapter
# peft=lora
adapter_reduction=6
# adapter_reduction=2
# lora_rank=4

#! Debug or not
debug=true
# debug=false
# ----------------

epochs=60
grad_accu=4
lr=5e-5
warmup_ratio=0.0
do_train=true
do_eval=true
# max_input=512
max_input=256
# max_input=1024
# max_target=128
max_target=256
# padding=true
num_beams=4
padding=

# function=normal #! add additional function
# function=normal_reduction_${adapter_reduction}
# function=normal_test_performance
# function=all_selfattn
function=all_crossattn
# function=onefnn
# function=twofnn
# function=allfnn
# function=sixfnn
# other_params=(".*11.layer.*.DenseReluDense.*"  ".*10.layer.*.DenseReluDense.*" ".*8.layer.*.DenseReluDense.*" ".*6.layer.*.DenseReluDense.*")
# other_params=(".*11.layer.*.DenseReluDense.*"  ".*10.layer.*.DenseReluDense.*")
# other_params=(".*11.layer.*.DenseReluDense.*")
# other_params=(".*.DenseReluDense.*")
# other_params=
# other_params=(".*SelfAttention.*")
other_params=(".*EncDecAttention.*")
exp=${now}_

task_type=unified_skg

#! Task -----------------------------------------
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
    extra_cmd="--skg_task"
    dataset=spider #! task
    dataset_config=$dataset
    template_name="none"
    echo Dataset: $dataset
fi

#! Debug-------------------------
if $debug
then
    # max_train_steps=500
    max_train_steps=5000
    echo DEBUGGING... 
    echo Max train steps: $max_train_steps
    extra_cmd+=" --debug"
    exp+="debug_"
    eval_step=50
    num_shots=10
else
    max_train_steps=20000
    echo FORMAL TRAINING
    echo Max train steps: $max_train_steps
    exp+="train_"
    num_shots=1000000
    eval_step=50
fi

#! Experiment----------------------
if $singularity
then
    # echo Use A100~
    echo Use Singularity V100~
    ds=deepspeed
    exp+='singds_'
else
    echo Use Local V100~
    # ds=/azure/yingxiu/ENVS/pet/bin/deepspeed
    # ds=deepspeed
    # ds=/opt/conda/envs/pet/bin/deepspeed
    ds=/home/v-yingxzhao/.conda/envs/peftxiu/bin/deepspeed
    exp+="localds_"
fi

# model_name=t5-base
# train_btz=4
# eval_btz=16
# ds_config=dsconfigs/t5base_peft.json # btz=4

model_name=t5-3b
train_btz=1
eval_btz=4
ds_config=dsconfigs/t53b_peft.json # btz=1
# exp+="v100_"
echo train btz: $train_btz
echo eval btz: $eval_btz

#! PEFT Module----------------------
if [[ -z $add_peft ]]
then
    echo Finetune all parameters... NO PEFT
    method=finetune
else
    echo Use PEFT: $peft
    method=$peft #! only tune PEFT
fi
echo Method: $method
model_path=/azure/yingxiu/DATA/${model_name}
# model_path=$model_name
# model_path="t5-3b"
# model_path=/azure/yingxiu/DATA/t5-3b
echo MODEL: $model_path

#? Experiment --------------------------------------------------------------
# exp=ds_data-${dataset}_${dataset_config}_ngpu-${ngpu}_model-${model_name}_run0
# exp=v100_${now}_${peft}_${dataset}_${dataset_config}_ngpu${ngpu}_model_${model_name}_run0
# exp+="${now}_${method}_${dataset}_${dataset_config}_ngpu${ngpu}_model_${model_name}_reduction${adapter_reduction}_run${runid}"
# exp+="${method}_${dataset}_${dataset_config}_ngpu${ngpu}_model_${model_name}_lorarank${lora_rank}_run${runid}"
# exp=date${now}_${peft}_${dataset}_${dataset_config}_ngpu${ngpu}_model_${model_name}_run1
exp+="${method}_${task_type}-task_${dataset}_ngpu${ngpu}_model_${model_name}"
exp+="_${function}_run${runid}" #! add other functions
echo EXP: $exp

#? Tasks configures -------------------------------------------------------
config=./configs/${method}.json+./configs/rte.json
# skg_cfg=META_TUNING/spider.cfg
skg_cfg=Salesforce/T5_base_finetune_spider_with_cell_value.cfg
# skg_cfg=Salesforce/T5_base_finetune_${dataset}.cfg
echo SKG-TASK: $skg_cfg

#? Output and Logs -----------------------------------------------------------
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
# $py -m torch.distributed.launch --nproc_per_node=$ngpu \
#ds=/mnt/users/yingxiu/pet/bin/deepspeed
# ds=deepspeed
# ds=/opt/conda/envs/pet/bin/deepspeed
# ds=/azure/yingxiu/ENVS/tzero/bin/deepspeed

export CUDA_VISIBLE_DEVICES=$gpuid
# $ds --include="worker-0:${gpuid}" --master_port=$port \
# $ds --include="worker-0:0,1,2,3,4,5,6,7" --master_port 22000 \
$ds \
    --num_gpus=$ngpu \
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
    --skg_cfg=$skg_cfg \
    --num_shots=$num_shots \
    --generation_num_beams=$num_beams \
    --other_params_list "${other_params[@]}" \
    -k exp_name=$exp lr=$lr warmup_ratio=$warmup_ratio exp_dir=$output adapter_reduction_factor=$adapter_reduction lora_rank=$lora_rank \
    ${extra_cmd} \
    >$log 2>&1

