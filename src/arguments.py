"""
File: arguments.py
- A file for dataclasses containing arguments
- UniQuanFArguments contains the hyperparameters of UniQuanF
- Reference:
    * https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class UniQuanFArguments:
    """
    Arguments pertaining to UniQuanF's hyperparameters
    """
    gradient_filtering: Optional[bool] = field(
        default=True,
        metadata={"help":"whether applying gradient filtering or not"},
    )
    u_lr: Optional[float] = field(
        default=1e-5,
        metadata={"help":"a learning rate for UQ's quantization parameters"},
    ) 
    b_lr: Optional[float] = field(
        default=1e-5,
        metadata={"help":"a learning rate for UQ's quantization parameters"},
    )
    period: Optional[int] = field(
            default=1,
            metadata={"help": "period for updating mappings"}
    )
    alternating_update_iters: Optional[int] = field(
            default=0,
            metadata={"help": "the number of iterations for initializing BCQ's quantization parameters"}
    )
    grid_search_iters: Optional[int] = field(
            default=100,
            metadata={"help": "the number of iterations for searching a clipping range"}
    )
    update_z: Optional[bool] = field(
        default=False,
        metadata={"help": "updating z or not"},
    )
    clipping_strategy: Optional[str] = field(
        default="min",
        metadata={"help": "Clipping strategy. min, max or balanced"},
    )
    mapping_function: Optional[str] = field(
        default="lpmapping",
        metadata={"help": "Clipping strategy. min, max or balanced"},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to quantize, fine-tune, train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    n_bits_w: Optional[int] = field(
        default=4,
        metadata={
            "help": (
                "bitwidth for weight quantization."
            )
        },
    )
    num_samples: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "size of the calibration dataset"
            )
        },
    )
    iters_w: Optional[int] = field(
        default=20000,
        metadata={"help": "number of iteration for optimization"},
    )
    group_size: Optional[int] = field(
        default=-1,
        metadata={"help":"quantization group-size. -1 represents channel-wise quantization"},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    n_bits_a: Optional[int] = field(
        default=32,
        metadata={
            "help": (
                "bitwidth for activation quantization."
            )
        },
    )
    channel_wise: Optional[bool] = field(
        default=False,
        metadata={"help": "apply channel_wise quantization for weights"},
    )
    disable_8bit_head_stem: Optional[bool] = field(
        default=False,
        metadata={"help": "disable_8bit_head_stem"},
    )
    keep_cpu: Optional[bool] = field(
        default=False,
        metadata={"help":'keep the calibration data on cpu'},
    )    
    weight: Optional[float] = field(
        default=0.01,
        metadata={
            "help": (
                "weight of rounding cost vs the reconstruction loss."
            )
        },
    )
    b_start: Optional[int] = field(
        default=20,
        metadata={
            "help": (
                "temperature at the beginning of calibration"
            )
        },
    )
    b_end: Optional[int] = field(
        default=2,
        metadata={
            "help": (
                "temperature at the end of calibration"
            )
        },
    )
    warmup: Optional[float] = field(
        default=0.2,
        metadata={
            "help": (
                "in the warmup period no regularization is applied"
            )
        },
    )
    init_wmode: Optional[str] = field(
        default='mse', metadata={"help": "init opt mode for weight"}
    )
    init_amode: Optional[str] = field(
        default='mse', metadata={"help": "init opt mode for activation"}
    )
    order: Optional[str] = field(
        default='before', metadata={"help":'order about activation compare to weight: before, after, together'}
    )
    prob: Optional[float] = field(
        default=1.0,
    )
    input_prob: Optional[float] = field(
        default=0.5,
    )
    quantization: Optional[bool] = field(
        default=True,
        metadata={"help":'quantization on/off'},
    )    
    symmetric: Optional[bool] = field(
        default=True,
        metadata={"help":'symmetric weight quantization ture/false, default true, for qdrop config is false'},
    )     
    a_lr: Optional[float] = field(
        default=0.,
        metadata={"help":'activation learning rate'},
    )   
    embedding_8bit: Optional[bool] = field(
        default=False,
        metadata={"help":'Whether use 8bit Embedding'},
    )   
    quantization_dataset: Optional[str] = field(
        default='train',
        metadata={"help":'zero-shot: select quantization source '},
    )
    save_model: Optional[str] = field(
        default=True,
        metadata={"help":'save_model directory '},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    recon_dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": 'datatype of reconstruction algorithm. fp32, bf16 is available and fp16 is not available.',
            "choices": ["bfloat16", "float32"],
        },
    )
    split_qkv: Optional[bool] = field(
        default=False,
        metadata={"help":'Split fused QKV weights'},
    )
    tp_size: Optional[int] = field(
        default=1,
        metadata={'help':'tensor parallel size'}
    )
    random_data_sample: Optional[bool] = field(
        default=False,
        metadata={"help":'random sampling from datasets'},
    )
    debug_mode: Optional[bool] = field(
        default=False,
        metadata={"help":'whether debug mode'},
    )
    in_ch_wise: Optional[bool] = field(
        default=False,
        metadata={"help":'whether transpse weights'},
    )
    quant_clip_search: Optional[bool] = field(
        default=True,
        metadata={"help":"where quantization clipping value search"},
    )
    lm_head_quant: Optional[bool] = field(
        default=False,
        metadata={"help":'whether quantize lm_head or not'},
    )
    use_bcq: Optional[bool] = field(
        default=True,
        metadata={"help":'whether use bcq or not'},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."
