import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from .adapters import T5LayerFFWithAdapter, T5LayerSelfAttentionWithAdapter, T5LayerCrossAttentionWithAdapter
from .lora import LoRALinear

def modify_with_adapter_lora(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():

        #* Add lora modules
        if re.fullmatch(config.lora_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.lora_layers, c_name):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    setattr(
                        module,
                        c_name,
                        LoRALinear(layer, config.lora_rank, config.lora_scaling_rank, config.lora_init_scale),
                    )

        #* Add adapter modules
        if re.fullmatch(".*block[.][0-9]*", m_name): 
            layer = nn.ModuleList()
            layer.append(module.layer[0])
            if module.is_decoder:
                layer.append(module.layer[1])
            layer.append(
                T5LayerFFWithAdapter(
                    module.layer[2] if module.is_decoder else module.layer[1],
                    config,
                    transformer.config,
                )
            )
            module.layer = layer
    return transformer
