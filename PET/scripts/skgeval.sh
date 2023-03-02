dataset=spider
skg_cfg=Salesforce/T5_base_finetune_spider_with_cell_value.cfg
# dataset=compwebq
# dataset=wikisql
# skg_cfg=Salesforce/T5_base_finetune_${dataset}.cfg

# eval_dir=/azure/yingxiu/DATA/PET/outputs/0219_train_local_adapter_unified_skg-task_spider_ngpu8_t5-base_normal_tbgrads_lr3e-3_run0/
# exp=0225_train_local_finetune_unified_skg-task_compwebq_ngpu4_t5-base_normal_trainbtz2_lr3e-3_run0
# exp=0226_train_local_finetune_unified_skg-task_wikisql_ngpu4_t5-base_normal_trainbtz2_lr3e-3_run0
# exp=0226_train_local_lora_unified_skg-task_wikisql_ngpu4_t5-base_layer68_fnn_lorarank6_trainbtz2_lr3e-3_run0
# exp=0226_train_local_lora_unified_skg-task_wikisql_ngpu4_t5-base_layer68_selfattn_lorarank6_trainbtz2_lr3e-3_run0
exp=0223_train_local_lora_unified_skg-task_spider_ngpu8_model_t5-base_layer0_selfattn_lorarank6_run0
# exp=0223_train_local_lora_unified_skg-task_spider_ngpu8_model_t5-base_layer2_selfattn_lorarank6_run0
eval_dir=/azure/yingxiu/DATA/PET/outputs/${exp}/
echo $eval_dir
tbdir=/azure/yingxiu/DATA/PET/tb_logs/$exp
mkdir -p $tbdir
# type=all
type=part

py=/home/v-yingxzhao/.conda/envs/peftxiu/bin/python

$py skg_evalall.py \
    --skg_cfg=$skg_cfg \
    --eval_dir=$eval_dir \
    --eval_type=$type \
    --tbdir=$tbdir