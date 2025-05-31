"""
File: alternating.py
- A file for implementing general alternating update
"""

import torch

def general_alternating_update(
            w, 
            qbits, 
            rounds=15, 
            group_size=-1, 
            transpose=False,  
            reshaped=False, 
            use_z=False,
            ):
    '''
    General alternating update to find alpha 
    rounds == 0: greedy algorithm
    rounds == 1: refined greedy algorithm
    rounds >= 2: alternating algorithm

    Args:
        w: a weight tensor of layer
        qbits: number of quantization bits for the `w`
        rounds: number of iterations for refining both alpha and B
        group_size: number of weights in which a scaling factor can be shared
        transpose: if `transpose` is True, `w` is a transposed when using this method.
        use_bst: if `use_bst` is True(default), the binary matrix is calculated using BST algorithm.
                    if `use_bst` is False, the binary matrix is calculated with greedy algorithm.
    Return:
        ret: a reconstructed (approximated) weight matrix
        B: a binary code matrix
        alpha: BCQ's scale factors
        z: a BCQ's shifting factor
    '''
    w_ = w.clone()
    w_ = w_.cuda()
    if not reshaped:
        if transpose:
            assert len(w_.shape) == 2, f'Check your weight shape {w_.shape}'
            w_ = w_.transpose(1, 0).contiguous()
        
        orig_shape = w_.shape
        group_size = group_size if group_size > 0 else orig_shape[-1]
        w_ = w_.view([-1, group_size])
 
    # init weighted
    ret, C, alpha = greedy_mean_torch(w_, n_bits=qbits)
    if use_z:
        z = torch.zeros_like(alpha)[:, :1]
    else:
        z = None
    if rounds > 0 and qbits > 1:
        for _ in range(rounds):
        # for _ in tqdm(range(rounds)):
            ret, C, alpha, z = refine_mean_torch(w_, 
                                                 C, 
                                                 alpha, 
                                                 z=z)
    if z is None:
        z = torch.zeros_like(alpha)[:, :1]
    if not reshaped:
        ret = ret.view(orig_shape) 
        if transpose:
            ret = ret.transpose(1, 0).contiguous()

        del w_

        C = C.reshape([orig_shape[0], orig_shape[1] // group_size, group_size, qbits])
        alpha = alpha.reshape([orig_shape[0], orig_shape[1] // group_size, qbits])
    torch.cuda.empty_cache()

    return ret, C, alpha, z

def greedy_mean_torch(w, n_bits=1):
    """
    Greedy initialization algorithm for BCQ scheme

    Args:
        w: a weight matrix
        n_bits: the number of bits

    Returns:
        _type_: _description_
    """
    C = torch.zeros(w.shape + (n_bits,), device=w.device)
    Alpha = torch.zeros(w.shape[0], n_bits, device=w.device)
    r, w_hat = w.clone(), 0.
    for i in range(n_bits):
        c = r.sign()
        
        # if wf is not None:
        #     a1sum = torch.sum(wf, dim=1)
        #     alpha = (r.abs()*wf).sum(dim=1) / torch.sum(wf, dim=1)
        #     alpha[torch.isnan(alpha)] = 0.
        #     alpha = alpha.view(alpha.shape[0], 1)
        # else:
        alpha = r.abs().mean(dim=1, keepdim=True)
        
        r -= c * alpha
        w_hat += c * alpha
        C[:,:,i] = c
        Alpha[:,i] = alpha.view(-1)
    
    del r, c, alpha
    torch.cuda.empty_cache()

    return w_hat, C, Alpha

def refine_mean_torch(w, C, Alpha, z=None):
    """
    Find proper binary code matrix C and BCQ's scale factors Alpha
    for weights w through alternaing update

    Args:
        w: target weight values
        C: a greedy-initialized binary coding matrix
        Alpha: a greedy-initialized BCQ's scale factors
        z: BCQ's shifting factor

    Returns:
        w_hat_new: an approximated weight during alternating update
        C_new: an updated binary code matrix
        Alpha_new: updated BCQ's scale factors
        z_new: an updated BCQ's shifting factors
    """
    w = w.float()
    d1, d2 = w.shape
    with torch.no_grad():
        n_bits = C.shape[-1]
        Ct = C.transpose(1, 2)
        C_cov = Ct.bmm(C)
        if z is not None:
            Ctw = Ct.bmm((w-z).unsqueeze(-1)).view(d1, n_bits)
        else:
            Ctw = Ct.bmm(w.unsqueeze(-1)).view(d1, n_bits)

        Alpha_new = batch_cg_torch(C_cov, Ctw, x=Alpha)
        Alpha_new, _ = Alpha_new.abs().sort(descending=True)
        if z is not None:
            C_new = fast_find_C_torch((w-z), Alpha_new)
        else:
            C_new = fast_find_C_torch(w, Alpha_new)
        w_hat_new = torch.einsum('ijl,il->ij', (C_new, Alpha_new))
        if z is not None:
            z_new = ((w - w_hat_new).mean(dim=-1)).unsqueeze(-1)
            w_hat_new += z_new
        else:
            z_new = None
    return w_hat_new, C_new, Alpha_new, z_new

def fast_find_C_torch(w, Alpha):
    """
    Efficiently find indices of corresponding quantization levels for weights
    by comparing distance between weights and quantization levels
    in parallel

    Args:
        w: weight values
        Alpha: BCQ's scale factors for determining quantization levels

    Returns:
        grid[idx]: found indices
    """
    n_bits = Alpha.shape[1]
    c = torch.tensor(data=[1, -1], dtype=w.dtype, device=w.device)
    grid = torch.cartesian_prod(*[c for _ in range(n_bits)])
    alpha_grid = (Alpha @ grid.T).unsqueeze(dim=-2)
    idx = torch.argmin((w.unsqueeze(-1) - alpha_grid).abs(), dim=-1)
    return grid[idx]

def batch_cg_torch(A, b, x=None):
    """
    Batch conjugate gradient for solving Ax = b

    Args:
        A: _description_
        b: _description_
        x: a vector of optimization targets

    Returns:
        _type_: _description_
    """
    
    d1, k, _ = A.shape
    # Initialize
    x = x.clone().view(d1, k, 1)
    b = b.view(d1, k, 1)
    r = b - A.bmm(x)
    rtr_new = r.transpose(1, 2).bmm(r)
    p = r.clone()
    # Perform batch CG
    for i in range(k):
        rtr = rtr_new
        Ap = A.bmm(p)
        alpha = rtr / (p.transpose(1, 2).bmm(Ap) + 1e-6)
        x += alpha * p
        r -= alpha * Ap
        rtr_new = r.transpose(1, 2).bmm(r)
        beta = rtr_new / (rtr + 1e-6)
        p = r + beta * p
    return x.view(d1, k)
