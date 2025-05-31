"""
File: evaluation.py
- A file for evaluating quantied models
- Reference:
    * https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
    * https://github.com/OpenGVLab/OmniQuant/blob/main/main.py
    * https://github.com/EleutherAI/lm-evaluation-harness/tree/main
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
# sys.path.append('_eval/mmlu_hf')
# sys.path.append('_eval/lm-evaluation-harness')
from evaluation_utils import run_evaluate as _eval_mmlu
from lm_eval import tasks, evaluator, utils
import os
from data_utils import get_loader_for_ppl
from tqdm import tqdm
import torch.nn as nn
import time
import gc

@torch.inference_mode
def eval_mmlu(args, model, tokenizer):
    """
    Evaluate 0-shot and 5-shot MMLU tasks

    Args:
        args: arguments for evaluation
        model: a model to evaluate
        tokenizer: a tokenizer
    """
    print("\n[Evaluate 0-shot and 5-shot MMLU tasks]")
    mmlu_save_dir = args.output_dir + "/mmlu"
    class MMLU_ARGS:
        def __init__(self, ntrain, data_dir, save_dir, model):
            self.ntrain = ntrain
            self.data_dir = data_dir
            self.save_dir = save_dir
            self.model = model

    zero_shot_mmlu_args = MMLU_ARGS(
        ntrain      = 0,
        data_dir    = args.mmlu_data,
        save_dir    = f'{mmlu_save_dir}/mmlu_0',
        model       = args.model
    )

    _eval_mmlu(args=zero_shot_mmlu_args,
               model=model,
               tokenizer=tokenizer,
               )
    
    five_shot_mmlu_args = MMLU_ARGS(
        ntrain      = 5,
        data_dir    = args.mmlu_data,
        save_dir    = f'{mmlu_save_dir}/mmlu_5',
        model       = args.model
    )

    _eval_mmlu(args=five_shot_mmlu_args,
               model=model,
               tokenizer=tokenizer,
               )
    print()

@torch.inference_mode
def eval_csr(args, model, tokenizer,
             task_list=['winogrande', 'piqa',
                        'hellaswag','arc_challenge',
                        'arc_easy','boolq'],
             ):
    """
    Evaluate 0-shot CommonSense Reasoning (CSR) tasks

    Args:
        args: arguments for evaluation
        model: a model to evaluate
        tokenizer: a tokenizer
        task_list: a list of tasks to evaluate
    """
    print("\n[Evaluate 0-shot CommonSense Reasoning (CSR) tasks]")
    csr_result_file = args.output_dir + "/csr.csv"

    task_manager = tasks.TaskManager(include_path='_eval/lm-evaluation-harness/lm_eval/tasks')
 
    task_names = task_manager.match_tasks(task_list)
    for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
    model_args = {'pretrained':model}

    results = evaluator.simple_evaluate(
        model='hf',
        model_args=model_args,
        tasks=task_list,
        num_fewshot=0,
        batch_size=8,
        max_batch_size=None,
        device='auto',
        use_cache=None,
        limit=None,
        # decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=False,
        gen_kwargs=None,
        task_manager=task_manager,
        verbosity = "ERROR",
    )

    results = results['results']
    _str = ''
    _tot_acc = 0.; _count = 0  
    print("- Evaluation results")
    for task in task_list:
        if 'acc_norm,none' in results[task]:
            _acc = results[task]['acc_norm,none']
        else:
            _acc = results[task]['acc,none']
        _str += f"{_acc*100:.4f},"
        print(f"  {task}: {_acc*100:.4f}")
        _tot_acc += _acc
        _count += 1
    _str += f"{_tot_acc/_count*100:.4f},"
    print(f"  Average: {_tot_acc/_count*100:.4f}")
    print()
    with open(csr_result_file, 'a') as f:
         f.write(_str + '\n')

@torch.inference_mode
def eval_gsm8k(args, model, tokenizer, batch_size=16,
             task_list=['gsm8k'],
             ):
    """
    Evaluate the GSM8k Accuracy

    Args:
        args: arguments for evaluation
        model: a model to evaluate
        tokenizer: a tokenizer
        batch_size: an evaluation batch size
        task_list: a list of tasks to evaluate (gsm8k)
    """
    print("\n[Evaluate GSM8k Accuracy]")
    gsm_result_file = args.output_dir + "/gsm.csv"
    task_manager = tasks.TaskManager(include_path='_eval/lm-evaluation-harness/lm_eval/tasks')
 
    task_names = task_manager.match_tasks(task_list)
    for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)

    model_args = {'pretrained':model}

    results = evaluator.simple_evaluate(
        model='hf',
        model_args=model_args,
        tasks=task_list,
        batch_size=batch_size,
        max_batch_size=None,
        device='auto',
        task_manager=task_manager,
        verbosity = "ERROR",
    )
    results = results['results']
    print(results)
    # _str = ''
    with open(gsm_result_file, 'a') as f:
         f.write(f"{results[task_list[0]]['exact_match,strict-match']*100:.4f},{results[task_list[0]]['exact_match_stderr,strict-match']*100:.4f}\n")
    

@torch.no_grad()
def eval_ppl(args, cache_dir, model, tokenizer, model_name):
    """
    Evaluate the ppls on WikiText2 and C4 benchmarks

    Args:
        args: arguments for evaluation
        cache_dir: a directory for save caches
        model: a model to evaluate
        tokenizer: a tokenizer
        model_name: the name of the model to evaluate
    """
    # Codes from OmniQuant 
    # https://github.com/OpenGVLab/OmniQuant/blob/main/main.py
    ppl_result_file = args.output_dir + "/ppl.csv"
    seqlen=2048
    for dataset in ['c4', 'wikitext2']:
        print(f"[Measuring ppl on {dataset}]")
        cache_path = f'{cache_dir}/{dataset}_{model_name}'
        print("- cache_path: ", cache_path)
        if os.path.exists(cache_path):
             print("- load cached loader")
             testenc = torch.load(cache_path)
        else:
            print("- Generate and save cached loader")
            testloader = get_loader_for_ppl(name=dataset, tokenizer=tokenizer)
            if 'c4' in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids
            torch.save(testenc, cache_path)
        nsamples = testenc.numel() // seqlen
        model.config.use_cache = False
        model.eval()
        nlls = []

        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to('cuda:0')
            outputs = model.model(batch)
            hidden_states = outputs[0]
            logits = model.lm_head(hidden_states)
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * seqlen) : 
                                   ((i + 1) * seqlen)][:, 1:
            ].to(model.lm_head.weight.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)


        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        print("*", dataset, ": ", ppl.item())
        with open(ppl_result_file, 'a') as f:
             f.write(f'{dataset},{ppl}\n')
        
if __name__ == "__main__":
    # Main file for running evaluation
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", "-t", nargs="+", 
                        help="Evaluation tasks in " + \
                        "['mmlu', 'csr', 'gsm-8k', 'wiki', 'c4']")
    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--mmlu_data", "-d", type=str, 
                        default="data/mmlu")
    parser.add_argument("--output_dir", "-s", type=str, 
                        default="./eval_results")
    parser.add_argument("--cache_dir", type=str, 
                        default="./cache")

    args = parser.parse_args()
    print(args, '\n')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load a model and a tokenizer
    model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')

    if 'mmlu' in args.tasks:
        eval_mmlu(args, model=model, tokenizer=tokenizer)

    if 'csr' in args.tasks:
        eval_csr(args, model=model, tokenizer=tokenizer,
                 task_list= ['winogrande', 'piqa','hellaswag',
                             'arc_challenge','arc_easy','boolq'])
        
    if 'gsm8k' in args.tasks:
        eval_gsm8k(args, model=model, tokenizer=tokenizer, batch_size=16,
                 task_list= ['gsm8k'])
        
    if 'ppl' in args.tasks:
        if "llama" in args.model.split('/')[-1].lower() or "mistral" in args.model.split('/')[-1].lower():
            model_name = args.model.split('/')[-1]
        else:
            model_name = args.model.split('/')[-2]
        print("Model Name: ", model_name)
        eval_ppl(args, cache_dir=args.cache_dir,
                model=model, tokenizer=tokenizer, model_name=model_name)
         
