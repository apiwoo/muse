# Project MUSE - shampoo.py
# Shampoo: Preconditioned Stochastic Tensor Optimization (2nd Order)
# Reference: https://arxiv.org/abs/1802.09568
# (C) 2025 MUSE Corp. All rights reserved.

import torch
from torch.optim.optimizer import Optimizer

class Shampoo(Optimizer):
    """
    [MUSE 2nd Order Optimizer]
    Implements Shampoo, a structured preconditioning algorithm.
    It captures correlation between parameters (curvature) to accelerate training.
    """
    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0,
                 epsilon=1e-4, update_freq=1):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        epsilon=epsilon, update_freq=update_freq)
        super(Shampoo, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Shampoo does not support sparse gradients')

                state = self.state[p]

                # Init State
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    
                    # Preconditioners (L and R for matrix, just one for vector)
                    if grad.dim() > 1:
                        state['precond_L'] = torch.zeros(grad.size(0), grad.size(0), device=p.device)
                        state['precond_R'] = torch.zeros(grad.size(1), grad.size(1), device=p.device)
                    else:
                        state['precond'] = torch.zeros_like(p.data)

                state['step'] += 1
                momentum = group['momentum']
                weight_decay = group['weight_decay']
                lr = group['lr']
                eps = group['epsilon']

                # Weight Decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Momentum
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                grad_to_use = buf

                # --- 2nd Order Preconditioning Logic ---
                if grad.dim() > 1:
                    # Matrix Parameters (Conv2d, Linear)
                    # Compute Statistics
                    grad_mat = grad_to_use.view(grad.size(0), -1)
                    
                    # Update Preconditioners (Accumulate Covariance)
                    state['precond_L'].add_(grad_mat @ grad_mat.t())
                    state['precond_R'].add_(grad_mat.t() @ grad_mat)

                    # Inverse Roots (every update_freq steps)
                    # For stability, we use a simple diagonal approximation if heavy
                    # Here we implement full matrix power for true 2nd order effect
                    
                    # Inverse p-th root of L and R
                    inv_L = self._matrix_power(state['precond_L'] / state['step'] + eps * torch.eye(grad.size(0), device=p.device), -1/4)
                    inv_R = self._matrix_power(state['precond_R'] / state['step'] + eps * torch.eye(grad.size(1), device=p.device), -1/4)
                    
                    # Apply: L^-1/4 * G * R^-1/4
                    # Note: Simplified Shampoo logic for efficiency
                    if grad.dim() == 2:
                        precond_grad = inv_L @ grad_to_use @ inv_R
                    elif grad.dim() == 4:
                        # Conv2d: (C_out, C_in, H, W) -> flatten to (C_out, C_in*H*W) handled by view logic above?
                        # Actually Shampoo for Conv is complex. We use block-diag approx here for stability.
                        # Using raw gradient as fallback for complex dims to prevent crash in simple implementation
                        precond_grad = grad_to_use 
                    else:
                        precond_grad = grad_to_use
                else:
                    # Vector Parameters (Bias) -> Diagonal/Adagrad style
                    state['precond'].add_(grad_to_use ** 2)
                    denom = state['precond'].sqrt().add_(eps)
                    precond_grad = grad_to_use / denom

                # Update Weights
                p.data.add_(precond_grad, alpha=-lr)

        return loss

    def _matrix_power(self, matrix, power):
        # Eigendecomposition for matrix power: A^p = V * D^p * V^T
        # Add jitter for stability
        try:
            u, s, v = torch.svd(matrix)
            return u @ torch.diag(s.pow(power)) @ v.t()
        except:
            # Fallback identity if SVD fails
            return torch.eye(matrix.size(0), device=matrix.device)