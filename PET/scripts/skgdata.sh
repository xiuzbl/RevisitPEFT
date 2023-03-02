# skg_cfg=Salesforce/T5_base_finetune_spider_with_cell_value.cfg
# skg_cfg=Salesforce/T5_base_finetune_sql2text.cfg
# skg_cfg=Salesforce/T5_base_finetune_sqa.cfg
skg_cfg=Salesforce/T5_base_finetune_wikisql.cfg
# skg_cfg=Salesforce/T5_base_finetune_wikitq.cfg
echo SKG TASK: $skg_cfg

py=/azure/yingxiu/ENVS/pet/bin/python
$py process_skg_data.py \
    --skg_cfg=$skg_cfg 