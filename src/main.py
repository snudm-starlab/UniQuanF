"""
File: main.py
- A main file for running UniQuanF (Unified Quantization with Flexible Mapping)
- Reference:
    * https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""

import json
import time
import sys
import logging
import os

import torch, datasets
import transformers
from transformers import (
    CONFIG_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from general_utils import *
from uniquanf import UniQuanF



# Define logger
logger = logging.getLogger("global")

def main():
    """
    The main file for running UniQuanF
    """
    # Parsing arguments
    uniquanf_args, model_args, data_args, training_args = parse_args()
    
    # Generate directories for outputs and logs
    os.makedirs(training_args.output_dir, exist_ok=True)
    os.makedirs(training_args.output_dir + '_log', exist_ok=True)

    # Set a logger
    datasets.utils.logging.set_verbosity(logging.ERROR)
    transformers.utils.logging.set_verbosity(logging.ERROR)
    set_logger(logger, training_args.output_dir + '_log')

    # Set a random seed for reproducibility
    set_seed(training_args.seed)
    
    # Get a pretrained model and a tokenizer
    model, tokenizer, torch_dtype = \
        load_model_tokenizer(model_args, training_args, logger)
    logger.info(f"* Load {model_args.model_name_or_path} done.")

    # Load dataloader
    quantization_source_dataloader = get_dataloader(model_args, 
                                                    training_args, 
                                                    data_args, 
                                                    tokenizer, 
                                                    logger)
    
    logger.info(f"* Get dataloader with {model_args.num_samples} data" +
                f"from {data_args.dataset_name}")

    # Build quantization parameters
    wq_params = {
        'n_bits'                    : model_args.n_bits_w, 
        'channel_wise'              : model_args.channel_wise, 
        'symmetric'                 : model_args.symmetric,
        'in_ch_wise'                : model_args.in_ch_wise,
        'group_size'                : model_args.group_size,
        'quant_clip_search'         : model_args.quant_clip_search,
        'clipping_strategy'         : uniquanf_args.clipping_strategy,
        'gradient_filtering'        : uniquanf_args.gradient_filtering,
        'period'                    : uniquanf_args.period,
        'alternating_update_iters'  : uniquanf_args.alternating_update_iters,
        'grid_search_iters'         : uniquanf_args.grid_search_iters,
        'mapping_function'          : uniquanf_args.mapping_function,
    } 

    aq_params = {
        'n_bits':       model_args.n_bits_a, # full precision
        'channel_wise': False, 
        'symmetric':    False,
        'input_prob':   model_args.input_prob,
    }
    recon_dtype = getattr(torch, model_args.recon_dtype)
    
    # Do quantization with UniQuanF
    uniquanf = UniQuanF(
        model       = model,
        data_loader = quantization_source_dataloader, 
        u_lr        = uniquanf_args.u_lr, 
        a_lr        = model_args.a_lr,
        b_lr        = uniquanf_args.b_lr, 
        iters       = model_args.iters_w, 
        num_samples = model_args.num_samples,
        wq_params   = wq_params,
        aq_params   = aq_params,
        batch_size  = training_args.per_device_train_batch_size, 
        torch_dtype = torch_dtype,
        recon_dtype = recon_dtype,
        input_prob  = model_args.input_prob,
        update_z    = uniquanf_args.update_z,
    )
    _st = time.time()
    uniquanf.minimize() # do quantization
    time_str = "\n************************************\n"
    time_str += f"-- Running time: {time.time()-_st}\n"
    time_str += "*************************************\n"
    logger.info(time_str)

    # save model
    if training_args.output_dir is not None and model_args.save_model:
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
    print(model)

if __name__ == "__main__":
    # Will error if the minimal version of Transformers is not installed. 
    # Remove at your own risks.
    check_min_version("4.21.0.dev0")

    require_version("datasets>=1.8.0", 
                    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
    main()
