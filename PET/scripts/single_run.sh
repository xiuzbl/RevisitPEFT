# gpuid=0,1,2,3,4,5,6,7,8,9,10,11
#gpuid=12,13,14,15
# ngpu=12
# a100=true
a100=

#gpuid=0,1,2,3
#gpuid=0,1,2,3,4,5,6,7
#gpuid=1,2,3,4,5,6,7
# gpuid=4,5,6,7
gpuid=8
ngpu=1
echo GPUID: $gpuid
echo Num_gpus: $ngpu

# exp=test01
# exp=test02
# exp=test03
#exp=test_t511b_run1
# exp=test_t511b_single_run0
exp=debug_wandb_run0
#exp=test_t5base
# exp=test_lora0
#exp=test_adapter04 # without deepspeed
# exp=test_adapter03
# exp=test_adapter02
# exp=test_bitfit0
#exp=occupy
#add_peft=true
add_peft=
peft=adapter

debug=true
# debug=
# debug_infer=true
debug_infer=

epochs=1
grad_accu=4
lr=1e-5
warmup_ratio=0.1
model_name=t5-base
# model_name=t5-3b
# model_name=t5-11b
model_path=/azure/yingxiu/DATA/${model_name}
do_train=true
do_eval=
dataset=super_glue
dataset_config=rte
template="does this imply"
train_btz=1
eval_btz=8
max_train_steps=5000
#max_train_steps=200
eval_step=50
max_input=1024
max_target=256
# padding=true
padding=

#! Experiment
# exp=${peft}_data${dataset_config}_ngpu${ngpu}_model_${model_name}_run0
echo EXP: $exp
config=./configs/adapter.json+./configs/rte.json

#* Output and Logs 
pet=/azure/yingxiu/DATA/PET
output=${pet}/outputs/$exp
log=${pet}/logs/$exp.log
tbdir=${pet}/tb_logs/$exp
cache_dir=${pet}/cache_dir/$exp
wandb_log=${pet}/wandblogs/$exp
mkdir -p ${pet}/cache_dir $cache_dir ${pet}/wandblogs ${wandb_log}
mkdir -p ${pet}/outputs $output ${pet}/logs ${pet}/tb_logs/$exp

# TODO: Training ..............................................................
py=/azure/yingxiu/ENVS/pet/bin/python
# ds=/azure/yingxiu/ENVS/pet/bin/deepspeed
# ds=/azure/yingxiu/ENVS/tzero/bin/deepspeed

export CUDA_VISIBLE_DEVICES=$gpuid
#$py -m torch.distributed.launch --nproc_per_node=$ngpu \
#torchrun --nproc_per_node=$ngpu \
$py run_single.py \
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
    --template_name "does this imply" \
    --max_train_steps=$max_train_steps \
    --gradient_accumulation_steps=$grad_accu \
    --debug \
    --eval_step=$eval_step \
    --max_length=$max_input \
    --target_max_length=$max_target \
    --pad_to_max_length=$padding \
    --add_peft=$add_peft \
    --wandb_log=$wandb_log \
    -k exp_name=$exp lr=$lr warmup_ratio=$warmup_ratio exp_dir=$output \
    >$log 2>&1
