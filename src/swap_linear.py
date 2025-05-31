"""
File: swap_linear.py
- A file for swapping linear layers into quantized ones
- Referece:
    * https://github.com/microsoft/LoRA
"""
import torch
import torch.nn as nn
import logging
logger = logging.getLogger("global")

from src.bcq_quant_layer import swapBCQLinear

def swap_quant_model(
    recon_block, 
    swap_type: str      = 'bcq',
    wq_params: dict      = None,
    aq_params: dict      = None,
):
    """
    Swap linear layers in a block into bcq_quant_layers

    Args:
        recon_block: a target block to swap
        swap_type: a target quantization scheme.
        wq_params: hyperparameters regarding quantization of weights
        aq_params: hyperparameters regarding quantization of activations
    """
    
    def _get_submodules(model, key):
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)

    with torch.no_grad():
        from transformers.pytorch_utils import Conv1D
        for name, layer in recon_block.named_modules():
            if (isinstance(layer, nn.Linear) and 
                'lm_head' not in name and 
                'project' not in name
            ):
                # logger.info(f'\nSwap {swap_type} linear with {name}')
                parent, target, target_name = _get_submodules(recon_block, name)
                if swap_type == 'bcq':
                    new_module = swapBCQLinear(layer, wq_params, aq_params) 
                else:
                    raise ValueError(f"invalid swap_type, {swap_type}")
                _replace_module(parent, target_name, new_module, target)

                torch.cuda.empty_cache()