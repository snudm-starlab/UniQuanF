"""
File: bcq_quant_layer.py
- A modified linear layer for quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from alternating import general_alternating_update
import time
import logging
logger = logging.getLogger("global")

def bcq_round_ste(inputs, 
                  alpha, 
                  bcq_shift, 
                  gradient_filtering=True, 
                  prev_idx=None, 
                  update_idx=True, 
                  update_idx_next=False,
                  mapping_function='lpmapping'):
    """
    Find quantization levels for each weight

    Args:
        inputs: an input (weights)
        alpha: BCQ's scale factors
        bcq_shift: a BCQ's shifting factor
        gradient_filtering: whether filtering gradient or not
        prev_idx: indicies of mapped quantization levels in the previous step
        update_idx: whether update indices or not
        update_idx_next: whether update indices in the next step or not
        mapping_function: the type of mapping function

    Returns:
        clamped_q_inputs + (clamped_inputs - clamped_inputs.detach()):
            mapped quantization levels after applying STE
        idx: indices of mapped quantization levels
    """

    # Initialization
    n_groups = alpha.shape[0]
    n_bits = alpha.shape[1]
    _inputs = (inputs - bcq_shift) - (2**n_bits - 1)/2
    alpha = alpha.to(device=_inputs.device, dtype=_inputs.dtype)

    b = torch.tensor(data=[1, -1], dtype=_inputs.dtype, device=_inputs.device)
    grid = torch.cartesian_prod(*[b for _ in range(n_bits)])
    alpha_grid = (alpha @ grid.T).unsqueeze(dim=-2)

    # Find indices of quantization levels
    if update_idx or (prev_idx is None):
        if mapping_function == 'direct' or prev_idx is None:
            # print("Do direct")
            idx = torch.argmin(
                (_inputs.unsqueeze(-1) - alpha_grid).abs(), 
                dim=-1)
        elif mapping_function == 'lpmapping':
            _indices = torch.tensor([-1, 0, 1], device=prev_idx.device)
            cand_idx = (prev_idx.unsqueeze(-1) + 
                        _indices).clip(0, 2**n_bits-1)
            w_r = alpha_grid.squeeze(dim=-2).gather(dim=-1, 
                            index=cand_idx.view(n_groups, -1).long()
                            ).view(n_groups,-1,_indices.shape[0])
            err_idx = torch.argmin(
                (_inputs.unsqueeze(-1)-w_r).abs(), dim=-1
                ).unsqueeze(-1)
            idx = cand_idx.gather(dim=-1, index=err_idx.long()).squeeze()
            del cand_idx, w_r, err_idx
        else:
            raise Exception("Unknown mapping function")
    else:
        idx = prev_idx
    
    # Find quantization levels using the found indices
    q_inputs = alpha_grid.squeeze(dim=-2).gather(dim=-1, index=idx)
    _thres = (torch.min(alpha.abs(), dim=-1).values).unsqueeze(1)

    # Gradient_filtering
    if gradient_filtering:
        clamped_q_inputs = torch.where((_inputs - q_inputs).abs() > _thres,
                                    q_inputs.detach(),
                                    q_inputs)
        clamped_inputs = torch.where((_inputs - q_inputs).abs() > _thres,
                                    inputs.detach(),
                                    inputs)
    else:
        clamped_q_inputs = q_inputs
        clamped_inputs = inputs
    clamped_q_inputs = clamped_q_inputs + bcq_shift + (2**n_bits - 1)/2

    if update_idx_next and mapping_function == 'direct':
        # remove idx if we remove idx in the next step
        idx = None    
    return clamped_q_inputs + (clamped_inputs - clamped_inputs.detach()), idx

def _lp_loss(pred, tgt, p=2.0, channel_wise=False):
    """
    loss function measured in L_p Norm
    """
    if 'tuple' in str(type(pred)):
        pred = pred[0]
    if 'tuple' in str(type(tgt)):
        tgt = tgt[0]
    
    x = (pred - tgt).abs().pow(p)
    
    if channel_wise:
        y = torch.flatten(x, 1)
        return y.mean(1)
    else:
        return x.mean()
    
def _quantize(
    x: torch.Tensor, 
    scale: torch.Tensor, 
    zero_point: torch.Tensor, 
    n_bits: int, 
    grid_search_iters: int = 0, 
    alternating_update_iters: int = 0,
    group_size: int =128,
):
    """
    Find new alpha and bcq_shift using the given scale and zero_point

    Args:
        x: an input for quantization (weights)
        scale: UQ's scale factor
        zero_point: UQ's zero-point
        n_bits: the number of bits
        grid_search_iters: 
            the number of iterations for a grid search (G)           
        alternating_update_iters: 
            the number of iterations for an alternating update (T)
        group_size: the size of weight groups

    Returns:
        x_dequant: a quantized weight value
        new_alpha: updated alpha
        bcq_shift: updated bcq_shift
    """
    
    # Transformation
    x_trans = (x/scale + zero_point) - (2**n_bits - 1)/2

    # general alternating update
    x_quant, _, new_alpha, bcq_shift = general_alternating_update(
        w=x_trans,
        qbits=n_bits,
        group_size=group_size,
        reshaped=True,
        rounds=alternating_update_iters,
        use_z = (grid_search_iters == 0),
    )

    # Detransformation
    x_dequant = (x_quant + (2**n_bits - 1)/2 - zero_point) * scale

    return x_dequant, new_alpha, bcq_shift

class BCQWeightQuantizer(nn.Module):
    """
    A class for quantizing a weight matrix
    """
    def __init__(
        self, 
        n_bits: int              = 8, 
        symmetric: bool          = False, 
        channel_wise: bool       = False, 
        org_weight: torch.Tensor = None, 
        in_ch_wise: bool         = False,
        group_size: int          = -1,
        quant_clip_search: bool  = True,
        gradient_filtering: bool           = True,
        period: int              = 1,
        alternating_update_iters:int       = 0,
        grid_search_iters: int    = 100,
        clipping_strategy: str   = 'min',
        mapping_function:str     = 'lpmapping',
    ):
        """
        An initialization function of BCQWeightQuantizer

        Args:
            n_bits: the number of bits for quantization
            symmetric: whether performing symmetric quantization or not
            channel_wise: whether performing channel-wise quantization or not
            org_weight: an original weight matrix
            in_ch_wise: whether performing in-channel-wise quantization or not
            group_size: the size of weight groups
            quant_clip_search: whether performing clipping range search or not
            gradient_filtering: whether performing gradient filtering or not
            period: a remapping period (p)
            alternating_update_iters:
                the number of iterations for an alternating update (T)
            grid_search_iters:
                the number of iterations for a grid search (G)
            clipping_strategy: a clipping strategy
            mapping_function: a type of mapping function.
        """
        super(BCQWeightQuantizer, self).__init__()
        self.symmetric = symmetric
        self.quant_clip_search = quant_clip_search
        self.grid_search_iters = grid_search_iters
        self.clip_search_iter = self.grid_search_iters # 100
        self.clipping_strategy = clipping_strategy
        self.gradient_filtering = gradient_filtering
        self.mapping_function = mapping_function

        assert 2 <= n_bits <= 8, f'Invalid Weight bits={n_bits}'
        self.n_bits = n_bits
        
        self.channel_wise = channel_wise
        self.eps = torch.tensor(1e-8, dtype=torch.float32)

        self.in_ch_wise = in_ch_wise
        if self.in_ch_wise:
            org_weight = org_weight.transpose(0, 1)

        self.group_size = group_size
        self.org_shape  = org_weight.shape
        if self.group_size != -1:
            org_weight = org_weight.reshape(-1, group_size)

        self.delta, self.zero_point, _alpha, bcq_shift \
            = self.unified_initialization(
                                    org_weight.detach(), 
                                    self.channel_wise,
                                    alternating_update_iters,
                                    group_size
                                    )
        
        self.alpha = nn.Parameter(_alpha, requires_grad=True)
        self.bcq_shift = bcq_shift
        self.idx = None
        self.count = 0
        self.period = period

        self.delta1 = nn.Parameter(torch.log(self.delta).detach())
        self.delta2 = nn.Parameter(torch.zeros_like(org_weight)) 
        self.delta3 = nn.Parameter(torch.zeros_like(self.delta))
        del self.delta

    def unified_initialization(self, x:torch.Tensor, 
                          channel_wise:bool = True,
                          alternating_update_iters:int = 0,
                          group_size:int = 128):
        """
        A unified initialization function for initializing quantization parameters from UQ and BCQ schemes

        Args:
            x: an input for quantization (weights)
            channel_wise: whether performing channel-wise quantization or not
            alternating_update_iters: 
                the number of iterations for an alternating update (T)
            group_size: the size of weight groups

        Returns:
            best_scale: an initialized UQ's scale factor
            best_zero_point: an initialized UQ's zero_point
            best_alpha: initialzed BCQ's scale factors
            best_bcq_shift: an initialized BCQ's shifting factor
        """
        if channel_wise:
            x = x.flatten(1)
        else:
            x = x.flatten(0).unsqueeze(0)

        # Initialization with min-max clipping range
        if self.symmetric: 
            # symmetric quantization
            best_scale = 2 * x.abs().max(1).values / (2**self.n_bits - 1)
            best_scale = torch.max(best_scale, self.eps).unsqueeze(1)
            best_zero_point = torch.zeros_like(best_scale)
        else: 
            # asymmetric quantization
            max_val, min_val = x.max(1).values, x.min(1).values
            best_scale = (max_val - min_val) / (2**self.n_bits - 1)
            best_scale = torch.max(best_scale, self.eps).unsqueeze(1)
            best_zero_point = (-1. * (min_val / best_scale.squeeze(1))).unsqueeze(1)

            best_alpha=torch.tensor([2.**(self.n_bits-2-i) for i in 
                                    range(self.n_bits)]).repeat(best_scale.shape[0], 1).to(best_scale.device)
            best_bcq_shift = torch.zeros_like(best_alpha)[:, :1]

        # Clipping range search using grid-search
        if self.quant_clip_search:
            tmp_scale = best_scale.clone()
            min_quant_error = torch.ones_like(tmp_scale) * 1e+8
            for i in range(1, self.clip_search_iter + 1):
                _ratio = \
                    (self.grid_search_iters - self.clip_search_iter + i) / self.grid_search_iters
                scale = tmp_scale * _ratio
                if self.symmetric:
                    zero_point = torch.zeros_like(scale)
                else:                    
                    if self.clipping_strategy == 'min':
                        zero_point = (-(min_val / 
                                        scale.squeeze(1))).unsqueeze(1)
                    elif self.clipping_strategy == 'max':
                        zero_point = (-((-1. * 
                                         (max_val - _ratio * 
                                          (max_val - min_val))) 
                                          / scale.squeeze(1)
                            )).unsqueeze(1)
                    else:
                        # balanced
                        zero_point = (-((-1. * (_ratio * min_val)) / scale.squeeze(1))).unsqueeze(1)
                
                # General alternating update
                x_dequant, alpha, bcq_shift = \
                        _quantize(x, 
                                  scale, 
                                  zero_point, 
                                  self.n_bits,
                                  self.grid_search_iters,
                                  alternating_update_iters,group_size=group_size)
                quant_error = _lp_loss(x, x_dequant, 
                                       2.4, self.channel_wise)\
                                       .view(-1, 1)

                # Update quantization parameters for weight groups
                # that exhibits reduced error
                best_scale = torch.where(quant_error < min_quant_error, 
                                         scale, best_scale)
                best_zero_point = torch.where(quant_error < min_quant_error,
                                              zero_point, best_zero_point)
                if alpha is not None:
                    best_alpha = torch.where(quant_error < min_quant_error,
                                             alpha, best_alpha)
                    best_bcq_shift = torch.where(quant_error < min_quant_error,
                                                 bcq_shift, best_bcq_shift)

                min_quant_error = torch.min(quant_error, min_quant_error)
        
        # Finalize the unified initialization
        x_dequant, best_alpha, best_bcq_shift = \
            _quantize(
                x, 
                scale=best_scale,
                zero_point = best_zero_point,
                n_bits = self.n_bits,
                grid_search_iters=self.grid_search_iters,
                alternating_update_iters = alternating_update_iters,
                group_size=group_size
            )
        return best_scale, best_zero_point, best_alpha, best_bcq_shift
    
    def forward(self, w: torch.Tensor):
        """
        Perform quantization

        Args:
            w: input weights

        Returns:
            w_dequant: quantized weights
        """

        if self.in_ch_wise:
            w = w.transpose(0, 1)
        
        if self.group_size != -1:
            w = w.reshape(-1, self.group_size)

        delta = (self.delta1 + self.delta2 + self.delta3).exp()
        scale = self.delta1.exp()

        # start quantization
        if False:
            raise NotImplementedError
        else:
            # Transformation
            w_tilde = w / delta + self.zero_point
            self.count += 1
            if self.period > 1:
                # Periodic remapping
                if self.count%self.period ==0 and \
                    self.mapping_function == 'direct':
                    self.idx = None
                    torch.cuda.empty_cache()
                w_int, self.idx = bcq_round_ste(w_tilde,
                                                self.alpha,
                                                self.bcq_shift,
                                                self.gradient_filtering,
                                                self.idx,
                                                (self.count%self.period==0),
                                                ((self.count+1)%self.period==0),
                                                mapping_function=
                                                    self.mapping_function
                                                )
            else:
                # Non-periodic remapping
                w_int, self.idx = bcq_round_ste(w_tilde,
                                         self.alpha,
                                         self.bcq_shift,
                                         self.gradient_filtering,
                                         None,
                                         True,
                                         False,
                                         mapping_function=
                                            self.mapping_function
                                        )
            # Detransformation
            w_dequant = (w_int - self.zero_point) * scale
        
        if self.group_size != -1:
            w_dequant = w_dequant.reshape(self.org_shape)

        if self.in_ch_wise:
            w_dequant = w_dequant.transpose(0, 1)

        return w_dequant

class BCQLinear(nn.Module):#
    """ 
        A class to support quantized linear functions in BCQ scheme
    """
    def __init__(
            self, 
            org_weight,
            bias,
            weight_quant_params: dict = {},
            act_quant_params: dict    = {},
            eval_mode: bool           = False,
    ):
        """
        Initialization function for BCQLinear

        Args:
            org_weight: the original weight
            bias: a bias
            weight_quant_params: 
                hyperparameters regarding quantization of weights
            act_quant_params: 
                hyperparameters regarding quantization of activations
            eval_mode: whether use evaluation mode or not
        """
        super().__init__()
        self.org_weight = org_weight
        self.weight = None
        if eval_mode:
            self.weight = nn.Parameter(
                torch.zeros(org_weight.size()), 
                requires_grad=False,
            )
            del self.org_weight

        self.use_weight_quant = True if weight_quant_params['n_bits'] < 16 else False
        self.use_act_quant    = False # True if act_quant_params['n_bits']    < 16 else False

        if weight_quant_params['n_bits'] < 16 and self.weight is None:
            self.weight_quantizer = BCQWeightQuantizer(
                **weight_quant_params, 
                org_weight = org_weight,
            )

        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        """
        A function for setting quantization state

        Args:
            weight_quant: the status of weight quantization
            act_quant: the status of activation quantization
        """
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        A forwarding function of BCQLinear

        Args:
            inputs: an input

        Returns:
            F.linear(inputs, weight, self.bias): an output
        """
        if self.use_weight_quant:
            if self.weight is None:
                weight = self.weight_quantizer(self.org_weight)
            else:
                weight = self.weight
        else:
            weight = self.org_weight
        
        # if self.use_act_quant:
        #     inputs = self.act_quantizer(inputs)
        
        return F.linear(inputs, weight, self.bias)
    
    def dequantize_weight(self):
        """
        Dequantize the weight matrix after optimization
        """
        weight = self.weight_quantizer(self.org_weight).contiguous()
        self.weight = nn.Parameter(weight.to(dtype=self.org_weight.dtype), requires_grad=False)

        del weight
        del self.org_weight
        del self.weight_quantizer

def swapBCQLinear(
        layer, 
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        eval_mode: bool = False,
):
    """
    Swap a linear layer into a BCQLinear 

    Args:
        layer: a linear layer to swap
        weight_quant_params: 
            hyperparameters regarding quantization of weights
        act_quant_params: 
            hyperparameters regarding quantization of activations
        eval_mode: whether use evaluation mode or not

    Returns:
        layer: a BCQLinear layer after swapping
    """
    weight = layer.weight

    if layer.bias is not None:
        bias = layer.bias
    else:
        bias = None

    layer = BCQLinear(
        org_weight = weight,
        bias = bias,
        weight_quant_params = weight_quant_params,
        act_quant_params = act_quant_params,
        eval_mode = eval_mode,
    )
    return layer
