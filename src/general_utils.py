"""
File: general_utils.py
- A file for general utility functions
- Reference:
    * https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""

import logging
import os, sys
from itertools import chain
from termcolor import colored

import torch
from torch.utils.data import DataLoader
import transformers
from transformers.testing_utils import CaptureLogger
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from datasets import load_dataset

from src.data_utils import get_dataset
from src.arguments import (
    UniQuanFArguments,
    ModelArguments, 
    DataTrainingArguments
)

def parse_args():
    """
    See all possible arguments in src/transformers/training_args.py
    or by passing the --help flag to this script.
    """
    parser = HfArgumentParser((UniQuanFArguments,
                               ModelArguments, 
                               DataTrainingArguments, 
                               TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        uniquanf_args, model_args, data_args, training_args = \
            parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        uniquanf_args, model_args, data_args, training_args = \
            parser.parse_args_into_dataclasses()
    return uniquanf_args, model_args, data_args, training_args


def set_logger(_logger, log_dir):
    """
    Setting the format of logger

    Args:
        _logger: a logger to set format
        log_dir: a directory for saving a log file
    """
    _logger.setLevel(logging.INFO)

    # Setup logger
    # Create formatter
    fmt = '[%(asctime)s %(name)s](%(filename)s %(lineno)d)|%(levelname)s| %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + \
                colored('|%(levelname)s|', 'blue') + ' %(message)s'

    # Create console handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S')
    )
    _logger.addHandler(console_handler)

    # Create file handlers
    file_handler_debug = logging.FileHandler(
        f'{log_dir}/opt.log', mode='a'
    )
    # Change here to set level for log file
    file_handler_debug.setLevel(logging.INFO)  
    file_handler_debug.setFormatter(
        logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    )
    _logger.addHandler(file_handler_debug)



def load_model_tokenizer(model_args, training_args, _logger):
    """
    Load pretrained model and tokenizer
    Args:
        model_args: arguments pertaining to models
        training_args: arguments pertaining to training
        _logger: a logger for logging

    Returns: 
        model: a loaded model
        tokenizer: a loaded tokenizer
        torch_dtype: the data type for inferencing the model
    """
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "attn_implementation": "eager", 
        "use_cache": False, 
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif  training_args.output_dir:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        _logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            _logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            _logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        _logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer, torch_dtype


def get_dataloader(model_args, training_args, data_args, tokenizer, _logger):
    """
    Generate a dataloader for optimization of UniQuanF
    
    Args:
        model_args: arguments pertaining to models
        training_args: arguments pertaining to training
        data_args: arguments pertaining to data
        tokenizer: a loaded tokenizer
        _logger: a logger to set format
    
    Returns:
        quantization_source_dataloader: a dataloader for optimization
    """
    raw_datasets = get_dataset(data_args, model_args)
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            _logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 2048
    else:
        if data_args.block_size > tokenizer.model_max_length:
            _logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)
    
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    with training_args.main_process_first(desc="grouping texts together"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    if 'train' in model_args.quantization_dataset:
        quantization_dataset = lm_datasets["train"]
    else:
        quantization_dataset = lm_datasets["validation"]
        
    if model_args.num_samples is not None:
        origin_len = len(quantization_dataset)
        max_eval_samples = min(origin_len, model_args.num_samples)
        quantization_dataset = quantization_dataset.select(range(max_eval_samples))

    quantization_source_dataloader = DataLoader(
        quantization_dataset, shuffle=False, 
        collate_fn=default_data_collator, 
        batch_size=training_args.per_device_train_batch_size)
    
    return quantization_source_dataloader