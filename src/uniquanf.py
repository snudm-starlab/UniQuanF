"""
File: uniquanf.py
- A file for optimization process of UniQuanF
- Reference:
    * https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import logging
# from typing import Optional
logger = logging.getLogger("global")

from src.loss import LossFunction
from src.cached_loader import CachedDataset, DataCacheWrapper
from src.bcq_quant_layer import BCQWeightQuantizer
from src.swap_linear import swap_quant_model

class UniQuanF:
    """
    A class for optimization process of UniQuanF.
    The "minimize" function contains the backbone code for optimization.
    """
    def __init__(
        self,
        model: torch.nn.Module, 
        data_loader: Dataset,
        wq_params: dict                  = None,
        aq_params: dict                  = None,
        batch_size: int                  = 16, 
        iters: int                       = 10000,
        input_prob: float                = 0.5,
        num_samples: int                 = 1024,
        u_lr: float                      = 4e-5, 
        a_lr: float                      = 4e-5,
        b_lr: float                      = 4e-5,
        torch_dtype                      = torch.float32,
        recon_dtype                      = torch.float32,
        update_z: bool                   = False,
    ):
        """
        Initialization function of UniQuanF class

        Args:
            model: a model to quantize
            data_loader: a dataloader contains sample data points
            wq_params: hyperparameters regarding quantization of weights
            aq_params: hyperparameters regarding quantization of activations
            batch_size: a batch size
            iters: the number of iterations for optimization
            input_prob: the probability for using quantized block's input
            num_samples: the number of sample data points
            u_lr: the learning rate for UQ's quantization parameters
            a_lr: the learning rate for quantization parameters regarding activations
            b_lr: the learning rate for BCQ's quantization parameters
            torch_dtype: the data type for inferencing model
            recon_dtype: the data type for optimization after the inference
            update_z: whether optimize the zero-point or not
        """
        self.wq_params   = wq_params
        self.aq_params   = aq_params
        self.model       = model.eval()
        self.data_loader = data_loader
        self.batch_size  = batch_size
        self.iters       = iters
        self.input_prob  = input_prob
        self.num_samples = num_samples
        self.u_lr        = u_lr
        self.a_lr        = a_lr
        self.b_lr        = b_lr
        self.torch_dtype = torch_dtype
        self.recon_dtype = recon_dtype
        self.update_z    = update_z
        
        self.use_weight_quant = True if wq_params['n_bits'] < 16 else False
        self.use_act_quant    = True if aq_params['n_bits'] < 16 else False

        self.attention_mask = None
        self.position_ids   = None
        self.qlayer_list = ['INTLinear', 'BCQLinear']

        # self.use_bcq = use_bcq

    def minimize(self, block_list_class=torch.nn.ModuleList):
        """
        Optimizing the quantization parameters of UniQuanF by
        minimizing reconstruction errors.

        Args:
            block_list_class: the class of the object containing 
                              the blocks in the model
        """
        # Find decoder layers
        for _, module in self.model.named_modules():
            if isinstance(module, block_list_class):
                block_units = module
                break

        block_units[0] = DataCacheWrapper(block_units[0])
        cached_data = CachedDataset(
            self.model, 
            self.data_loader, 
            self.input_prob, 
            self.num_samples,
            block_list_class
        )
        block_units[0] = block_units[0].module
        
        for idx in tqdm(range(len(block_units))):
            _str = '\n' + '='*60 + '\n'
            _str += f'    Layer {idx} Optimization Start\n' 
            _str += '='*60 
            logger.info(_str)
            
            # 1. Full-precision model Activation Caching
            if idx > 0 :
                fp_block = block_units[idx]           
                wrapped_block = DataCacheWrapper(fp_block)
                cached_data.fp_data_caching(wrapped_block)
            cached_data.cached_fp_input = [_fp_input.cpu() for _fp_input in cached_data.cached_fp_input]

            cached_dataloader = DataLoader(
                cached_data,
                shuffle = True,
                batch_size = 1,
            )

            # 2. Make independent Block
            recon_block = block_units[idx].to('cuda')

            # 3. Quantize independent Block
            swap_quant_model(
                recon_block,
                swap_type = 'bcq',
                wq_params  = self.wq_params,
                aq_params  = self.aq_params,
            )
            # 4. Block Minimization Reconstruction Error
            recon_block = self.type_cast(recon_block, self.recon_dtype)
            recon_block = self.blockReconstruction(
                recon_block, 
                cached_dataloader,
            )

            # 5. Dequantize Block
            recon_block = self.type_cast(recon_block, self.torch_dtype)
            self.dequantize_block(recon_block)

            # 6. Quantized model Activation caching
            wrapped_block = DataCacheWrapper(recon_block)
            cached_data.q_data_caching(wrapped_block)
            # Move q_data to cpu to avoid OOM
            cached_data.cached_q_input = [_q_input.cpu() for _q_input in cached_data.cached_q_input]
            
            block_units[idx] = recon_block

            torch.cuda.empty_cache()
        
        del cached_data
        del cached_dataloader
        torch.cuda.empty_cache()

    def type_cast(self, block, cast_type):
        """
        A function for casting the data type of a block

        Args:
            block: a Transformer block
            cast_type: a data type for casting

        Returns:
            block: a block after type casting
        """
        if block is None:
            return None

        if cast_type == torch.float32:
            block = block.float()
        elif cast_type == torch.float16:
            block = block.half()
        elif cast_type == torch.bfloat16:
            block = block.bfloat16()
        else:
            print(cast_type)
            raise ValueError("Invalid Type cast")
        return block

    def set_quant_state(self, block):
        """
        Setting the quantization state of modules in a block

        Args:
            block: a Transformer block
        """
        for _, module in block.named_modules():
            if module.__class__.__name__ in self.qlayer_list:
                module.set_quant_state(self.use_weight_quant, self.use_act_quant)

    def dequantize_block(self, block):
        """
        Dequantize the block after optimization

        Args:
            block: a Transformer block
        """
        for _, module in block.named_modules():
            if module.__class__.__name__ in self.qlayer_list:
                module.dequantize_weight()

    def blockReconstruction(self, recon_block:nn.Module, cached_dataloader):
        """
        Optimize and quantize a block

        Args:
            recon_block: a block to optimize and quantize
            cached_dataloader: a cached data loader

        Returns:
            _type_: _description_
        """

        u_para = []
        a_para = []
        b_para = []

        # Collect Weight, Activation Quantization params
        num_delta = 3 
        for name, module in recon_block.named_modules():

            if isinstance(module, BCQWeightQuantizer):
                u_para += [getattr(module, 'delta' + str(idx+1)) for idx in range(num_delta)]
                b_para += [module.alpha]

        all_params = [ {'params': u_para, 'lr': self.u_lr}, 
                      {'params': a_para, 'lr': self.a_lr},
                      {'params': b_para, "lr": self.b_lr}]

        optimizer = torch.optim.Adam(all_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.iters, eta_min=0.)
        
        # Define a loss function
        loss_func = LossFunction(
            recon_block, 
            max_count = self.iters, 
        )

        epochs = int(self.iters/len(cached_dataloader))
        remainder = self.iters - len(cached_dataloader) * epochs

        for name, param in recon_block.named_parameters():
            if 'quantizer' not in name:
                param.requires_grad = False

        # Optimization
        optimizer.zero_grad()
        for epoch in tqdm(range(epochs+1)):
            for step, batch in enumerate(cached_dataloader):
                if epoch == epochs and step == remainder:
                    break

                # Make input, output for reconstruction error
                input_q   = batch['q_input'].squeeze(0).cuda()
                input_fp  = batch['fp_input'].squeeze(0).cuda()
                output_fp = batch['fp_output'].squeeze(0)
                
                self.attention_mask = batch.get('attention_mask', None)
                self.position_ids   = batch.get('position_ids', None)
                cache_position      = batch.get('cache_position', None)

                if self.attention_mask is not None:
                    self.attention_mask = self.attention_mask.squeeze(0)
                if self.position_ids is not None:
                    self.position_ids = self.position_ids.squeeze(0)
                    cache_position = cache_position.squeeze(0)
                
                if self.input_prob < 1.0:
                    input_q = torch.where(
                        torch.rand_like(
                            input_q, dtype=input_q.dtype) < self.input_prob, input_q, input_fp)
                
                # type cast for fp32 reconstruction
                input_q             = self.type_cast(input_q, self.recon_dtype)
                self.attention_mask = self.type_cast(self.attention_mask, self.recon_dtype)

                if self.attention_mask is not None: # language task
                    if self.position_ids is not None: # for llama, gemma
                        output_q = recon_block(
                            hidden_states  = input_q, 
                            attention_mask = self.attention_mask, 
                            position_ids   = self.position_ids,
                            cache_position = cache_position,
                        )
                    else:
                        output_q = recon_block(
                            hidden_states  = input_q, 
                            attention_mask = self.attention_mask,
                        )
                else:
                    output_q = recon_block(input_q)

                loss = loss_func(output_q, output_fp)
                loss.backward()
                
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()

        del optimizer
        torch.cuda.empty_cache()
        return recon_block
