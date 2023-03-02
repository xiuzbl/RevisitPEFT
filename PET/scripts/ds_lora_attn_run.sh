# a100=false
# a100=true
singularity=false
# singularity=true
port=6000
# port=$1
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
# peft=adapter
peft=lora
adapter_reduction=6
# lora_rank=4
# lora_rank=16
lora_rank=6
# lora_rank=32
# lora_rank=24
# lora_rank=20
# lora_rank=12
echo lora_rank $lora_rank

#! Debug or not
debug=true
# debug=false
#! ----------------

epochs=60
grad_accu=4
# lr=5e-5
lr=3e-3
warmup_ratio=0.0
do_train=true
do_eval=true
task_type=unified_skg

# function=normal #! add additional function
# function=normal_lr${lr}
# function=normal_tbgrads_lorarank${lora_rank}
# function=normal_lorarank${lora_rank}
# function=allselfattn_tbgrads_lorarank${lora_rank}
# function=fourselfattn_tbgrads_lorarank${lora_rank}
# layer=0
# layer=2
# layer=8
# layer=$3
# layer=1011
# layer=68
layer=23
function=layer${layer}_selfattn_lorarank${lora_rank}
# function=layer${layer}_fnn_lorarank${lora_rank}
# function=twolayer${layer}_fnn_lorarank${lora_rank}_trainbtz${train_btz}
# function=twofnn
# function=allfnn
# function=allfnn_closeepo10_norecover
# function=debug
# other_params=(".*${layer}.layer.*.DenseReluDense.*")
# other_params=(".*6.layer.*.DenseReluDense.*"  ".*8.layer.*.DenseReluDense.*")
# other_params=(".*.DenseReluDense.*") # tune all fnn layers
# other_params=(".*23.layer.*.DenseReluDense.*") 
# other_params=
# other_params=(".*11.layer.*.SelfAttention.*"  ".*10.layer.*.SelfAttention.*" ".*9.layer.*.SelfAttention.*" ".*8.layer.*.SelfAttention.*")
# other_params=(".*6.layer.*.SelfAttention.*"  ".*8.layer.*.SelfAttention.*")
# other_params=(".*11.layer.*.SelfAttention.*"  ".*10.layer.*.SelfAttention.*" ".*9.layer.*.SelfAttention.*" ".*8.layer.*.SelfAttention.*")
# other_params=(".*11.layer.*.SelfAttention.*"  ".*10.layer.*.SelfAttention.*" ".*9.layer.*.SelfAttention.*" ".*8.layer.*.SelfAttention.*" ".*7.layer.*.SelfAttention.*")
other_params=(".*${layer}.layer.*.SelfAttention.*")
echo Other parameters "${other_params[@]}"
exp=${now}_
close_epoch=-1
# close_epoch=10
early_stop_steps=5000
# tb_grads=true
tb_grads=

# max_input=512
max_input=256
# max_input=1024
# max_target=256
max_target=128
# padding=true
num_beams=1
padding=

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
    max_train_steps=500
    # max_train_steps=5000
    echo DEBUGGING... 
    echo Max train steps: $max_train_steps
    extra_cmd+=" --debug"
    exp+="debug_"
    eval_step=20
    num_shots=10000000
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
    ds=/home/v-yingxzhao/.conda/envs/peftxiu/bin/deepspeed
    # ds=deepspeed
   # exp+="v100_"
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
exp+="${method}_${task_type}-task_${dataset_config}_ngpu${ngpu}_model_${model_name}"
exp+="_${function}_run${runid}" #! add other functions
echo EXP: $exp

#? Tasks configures -------------------------------------------------------
config=./configs/${method}.json+./configs/rte.json
# skg_cfg=META_TUNING/spider.cfg
skg_cfg=Salesforce/T5_base_finetune_spider_with_cell_value.cfg
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
# ds=/mnt/users/yingxiu/pet/bin/deepspeed
# py=/opt/conda/envs/pet/bin/python
# ds=/opt/conda/envs/pet/bin/deepspeed
# ds=deepspeed
# ds=/azure/yingxiu/ENVS/tzero/bin/deepspeed
# $py -m torch.distributed.launch --nproc_per_node=$ngpu \

export CUDA_VISIBLE_DEVICES=$gpuid
# $ds --include="worker-0:${gpuid}" --master_port=$port \
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
    --close_fnn_epochs=$close_epoch \
    --early_stop_steps=$early_stop_steps \
    --tb_grads=$tb_grads \
    -k exp_name=$exp lr=$lr warmup_ratio=$warmup_ratio exp_dir=$output adapter_reduction_factor=$adapter_reduction lora_rank=$lora_rank \
    ${extra_cmd} \
    >$log 2>&1
