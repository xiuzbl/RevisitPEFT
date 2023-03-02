# a100=true
# a100=false
# singularity=false
singularity=true
port=1120
# port=1234
# port=50000
echo PORT: $port
now="$(date +'%m%d')"
echo DATE: $now
# now=0128

# runid=01
# runid=1a
runid=0
#gpuid=0,1,2,3
# gpuid=0
gpuid=0,1,2,3,4,5,6,7
# gpuid=16
# gpuid=8,9,10,11,12,13,14,15
# gpuid=4,5,6,7
ngpu=8
# ngpu=1
echo GPUID: $gpuid
echo Num_gpus: $ngpu

add_peft=true
# add_peft=
peft=adapter
# peft=lora
# adapter_reduction=6
adapter_reduction=12
lora_rank=4
echo adapter_reduction_ratio $adapter_reduction

#! Debug or not
# debug=true
debug=false
# ----------------

epochs=60
# grad_accu=4
grad_accu=1
# lr=1e-5
# lr=5e-5
lr=3e-3
echo learning rate $lr
# warmup_ratio=0.1
warmup_ratio=0.0

model_name=t5-base
# model_name=t5-3b
# model_name=t5-11b
do_train=true
# do_train=
do_eval=true
# do_eval=
# task_type=t0
task_type=unified_skg

train_btz=4
eval_btz=8
echo train btz: $train_btz
echo eval btz: $eval_btz
# function=normal #! add additional function
# function=normal_lr${lr}
# function=twofnn
# function=allfnn
# function=allfnn_closeepo10_norecover
# function=debug
function=all_selfattn
# other_params=(".*11.layer.*.DenseReluDense.*"  ".*10.layer.*.DenseReluDense.*")
# other_params=(".*.DenseReluDense.*") # tune all fnn layers
# other_params=
other_params=(".*SelfAttention.*")
# other_params=(".*EncDecAttention.*")
# exp=random_init_${now}_
# exp=dense.wi_${now}_
exp=${now}_
close_epoch=-1
# close_epoch=10

# max_input=512
max_input=256
# max_input=1024
max_target=256
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
    dataset_config=spider
    template_name="none"
    echo Dataset: $dataset
fi

#! Debug-------------------------
# if [[ -z "$debug" ]]
if $debug
then
    # max_train_steps=50000
    max_train_steps=500
    echo DEBUGGING... 
    echo Max train steps: $max_train_steps
    extra_cmd+=" --debug"
    exp+="debug_"
    eval_step=20
    # num_shots=10
    num_shots=10000000
    # exp=debug_wandb_
else
    max_train_steps=20000
    # max_train_steps=5000
    # max_train_steps=2500
    # max_train_steps=10000
    echo FORMAL TRAINING
    echo Max train steps: $max_train_steps
    exp+="train_"
    num_shots=1000000
    eval_step=50
    # eval_step=100
fi

#! Experiment----------------------
if $singularity
then
    # echo Use A100~
    echo Use Singularity V100~
    # ds=deepspeed
    py=python
    exp+='sing_'
else
    echo Use Local V100~
    # ds=/azure/yingxiu/ENVS/pet/bin/deepspeed
    # ds=deepspeed
    # ds=/opt/conda/envs/pet/bin/deepspeed
    # py=/opt/conda/envs/pet/bin/python
    py=/home/v-yingxzhao/.conda/envs/peftxiu/bin/python
    exp+="local_"
fi

#! PEFT Module----------------------
if [[ -z $add_peft ]]
then
    echo Finetune all parameters... NO PEFT
    method=finetune
else
    echo Use PEFT: $peft
    method=$peft #! only tune PEFT
    # method=finetune_$peft # tune the whole model with PEFT
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
# exp=0128_finetune_unified_skg-task_spider_ngpu8_model_t5-base_run0
exp+="${method}_${task_type}-task_${dataset_config}_ngpu${ngpu}_model_${model_name}_lr${lr}_reduction${adapter_reduction}"
exp+="_${function}_run${runid}" #! add other functions
echo EXP: $exp

#? Tasks configures------------------------------------------------------
config=./configs/${method}.json+./configs/rte.json
# skg_cfg=META_TUNING/spider.cfg
skg_cfg=Salesforce/T5_base_finetune_spider_with_cell_value.cfg
echo SKG-TASK: $skg_cfg

#? Output and Logs -----------------------------------------------------------
pet=/azure/yingxiu/DATA/PET
output=${pet}/outputs/$exp
log=${pet}/logs/$exp.log
tbdir=${pet}/tb_logs/$exp
cache_dir=${pet}/cache_dir/$exp
mkdir -p ${pet}/cache_dir $cache_dir
mkdir -p ${pet}/outputs $output ${pet}/logs ${pet}/tb_logs/$exp

# TODO: Training ..............................................................
# py=/azure/yingxiu/ENVS/pet/bin/python
# py=python
# ds=/azure/yingxiu/ENVS/pet/bin/deepspeed
# ds=/azure/yingxiu/ENVS/tzero/bin/deepspeed

export CUDA_VISIBLE_DEVICES=$gpuid
# $py \
$py -m torch.distributed.launch --nproc_per_node=$ngpu --master_port=$port \
    run.py \
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
    --skg_cfg=$skg_cfg \
    --num_shots=$num_shots \
    --generation_num_beams=$num_beams \
    --other_params_list "${other_params[@]}" \
    --close_fnn_epochs=$close_epoch \
    -k exp_name=$exp lr=$lr warmup_ratio=$warmup_ratio exp_dir=$output adapter_reduction_factor=$adapter_reduction lora_rank=$lora_rank \
    ${extra_cmd} \
    >$log 2>&1
