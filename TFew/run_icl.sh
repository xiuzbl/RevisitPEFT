gpuid=8
exp_name=icl_rte
model=t5-base
dataset=data/few_shot/rte
log=$exp_name.log

py=/azure/yingxiu/ENVS/pet/bin/python
# py=/azure/yingxiu/ENVS/tfew/bin/python

CUDA_VISIBLE_DEVICES=$gpuid \
$py src/ticl/test_icl.py \
    --pretrained_model=$model \
    --dataset=$dataset \
    >$log 2>&1
