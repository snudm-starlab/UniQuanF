"""
File: data_utils.py
- A file for loading and preprocessing data
- Refence:
    * https://github.com/OpenGVLab/OmniQuant/blob/main/datautils.py
"""

from datasets import load_dataset
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import torch
import random

def get_dataset(data_args, model_args):
    """
    Get a dataset for quantization

    Args:
        data_args: arguments pertaining to data
        model_args: arguments pertaining to models

    Returns:
        raw_datasets: loaded dataset before preprocessing
    """
    if data_args.dataset_name is not None:
        if 'c4' in data_args.dataset_name:
            raw_datasets = load_dataset(
                'allenai/c4',
                data_files={'train': 
                            'en/c4-train.00000-of-01024.json.gz'},
                cache_dir=model_args.cache_dir,
                token=True if model_args.use_auth_token else None,
            )
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    'allenai/c4',
                    data_files={'validation': 
                                'en/c4-validation.00000-of-00008.json.gz'},
                    split=f"validation[:1%]",
                    cache_dir=model_args.cache_dir,
                    token=True if model_args.use_auth_token else None,
                )
                raw_datasets["train"] = load_dataset(
                    'allenai/c4',
                    data_files={'train': 
                                'en/c4-train.00000-of-01024.json.gz'},
                    split=f"train[:1%]",
                    cache_dir=model_args.cache_dir,
                    token=True if model_args.use_auth_token else None,
                )

        elif 'gsm8k' in data_args.dataset_name:
            raw_datasets = load_dataset(
                "openai/gsm8k", 
                name='main', 
                data_files=None,
                cache_dir=model_args.cache_dir,
                )

            for _split in ["train", "test"]:
                _raw_dataset = raw_datasets[_split]
                text = []
                for _item in _raw_dataset:
                    _q, _a = _item["question"], _item["answer"]
                    _text = f"Question: {_q}\nAnswer: {_a.split('### ')[-1].rstrip()}"
                    text.append(_text)
                _raw_dataset = _raw_dataset.add_column("text", text)
                _raw_dataset = _raw_dataset.remove_columns(["question", "answer"])
                
                raw_datasets[_split] = _raw_dataset
            temp_ds = raw_datasets["train"].train_test_split(
                            test_size=\
                                data_args.validation_split_percentage/100, 
                            seed=42)
                        
            raw_datasets["train"]      = temp_ds["train"]
            raw_datasets["validation"] = temp_ds["test"]        
        else:
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                token=True if model_args.use_auth_token else None,
            )

            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=True if model_args.use_auth_token else None,
                )

                raw_datasets["train"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    token=True if model_args.use_auth_token else None,
                )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage 
        # will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            
    return raw_datasets

# Get datasets for evaluating ppls
def get_wikitext2_for_ppl(tokenizer):
    """
        Get WikiText2 dataset for evaluating ppl

    Args:
        tokenizer: a tokenizer

    Returns:
        testenc: a tokenized dataset for evaluating ppl
    """
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')    
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    return testenc

def get_c4_for_ppl(seqlen, tokenizer):
    """
        Get C4 dataset for evaluating ppl

    Args:
        seqlen: a sequence length for evaluating ppl
        tokenizer: a tokenizer

    Returns:
        valenc: a tokenized dataset for evaluating ppl
    """
    valdata = load_dataset(
        'allenai/c4', 
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    return valenc

def get_loader_for_ppl(name, seqlen=2048, tokenizer=''):
    """
        Get a dataset for evaluating ppl

    Args:
        name: the name of benchmark dataset
        seqlen: a sequence length for evaluating ppl
        tokenizer: a tokenizer

    Returns:
        a tokenized dataset for evaluating ppl
    """
    if 'wikitext2' in name:
        return get_wikitext2_for_ppl(tokenizer)
    elif 'c4' in name:
        return get_c4_for_ppl(seqlen, tokenizer)
    



