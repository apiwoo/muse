# Project MUSE - shampoo.py
# Shampoo: Preconditioned Stochastic Tensor Optimization (2nd Order)
# Reference: https://arxiv.org/abs/1802.09568
# [Fixed] Correct dimension handling for Convolutional Layers (Flattened Matrix View)
# [Optimization] Lazy Update (update_freq=10) with Stability Fixes (NaN Guard, SVD Clamping)
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
                 epsilon=1e-4, update_freq=10):  # [Config] Default updated to 10
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        epsilon=epsilon, update_freq=update_freq)
        super(Shampoo, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            update_freq = group['update_freq']
            lr = group['lr']
            eps = group['epsilon']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
               
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Shampoo does not support sparse gradients')

                # [Safety] Check for NaN/Inf in gradients
                # If gradient is broken, skip this parameter update to save the model
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    continue

                state = self.state[p]

                # [Fix] Pre-calculate flattened matrix view for dimension checks
                # Conv2d (C_out, C_in, H, W) -> (C_out, C_in * H * W)
                if grad.dim() > 1:
                    grad_mat = grad.view(grad.size(0), -1)
                else:
                    grad_mat = None

                # Init State
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                   
                    # Preconditioners (L and R for matrix, just one for vector)
                    if grad.dim() > 1:
                        # [Critical Fix] Use grad_mat size instead of grad.size(1)
                        state['precond_L'] = torch.zeros(grad_mat.size(0), grad_mat.size(0), device=p.device)
                        state['precond_R'] = torch.zeros(grad_mat.size(1), grad_mat.size(1), device=p.device)
                        
                        # Cache storage for inverse matrices
                        state['inv_L'] = torch.eye(grad_mat.size(0), device=p.device)
                        state['inv_R'] = torch.eye(grad_mat.size(1), device=p.device)
                    else:
                        state['precond'] = torch.zeros_like(p.data)

                state['step'] += 1

                # Weight Decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Momentum
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                grad_to_use = buf

                # --- 2nd Order Preconditioning Logic ---
                if grad.dim() > 1:
                    # Update Preconditioners (Accumulate Covariance)
                    grad_mat = grad_to_use.view(grad.size(0), -1)
                   
                    state['precond_L'].add_(grad_mat @ grad_mat.t())
                    state['precond_R'].add_(grad_mat.t() @ grad_mat)

                    # [Optimization] Lazy Update Logic
                    # SVD is expensive, so only run it every 'update_freq' steps.
                    if state['step'] == 1 or state['step'] % update_freq == 0:
                        # Inverse p-th root of L and R
                        # Add epsilon directly to diagonal inside matrix_power for better control
                        inv_L = self._matrix_power(state['precond_L'] / state['step'], -1/4, eps)
                        inv_R = self._matrix_power(state['precond_R'] / state['step'], -1/4, eps)
                        
                        # Cache the results
                        state['inv_L'] = inv_L
                        state['inv_R'] = inv_R
                    else:
                        # Use cached preconditioners
                        inv_L = state['inv_L']
                        inv_R = state['inv_R']
                   
                    # Apply: L^-1/4 * G * R^-1/4
                    if grad.dim() == 2:
                        precond_grad = inv_L @ grad_to_use @ inv_R
                    elif grad.dim() == 4:
                        precond_mat = inv_L @ grad_mat @ inv_R
                        precond_grad = precond_mat.view_as(grad_to_use)
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

    def _matrix_power(self, matrix, power, eps=1e-4):
        # [Stability] Force Float32 for SVD
        # SVD on FP16 is very unstable and causes NaNs easily.
        matrix_f32 = matrix.float()
        
        # Add Epsilon to diagonal for numerical stability (Regularization)
        # Prevents zero eigenvalues
        matrix_f32 += eps * torch.eye(matrix.shape[0], device=matrix.device)

        try:
            # Eigendecomposition: A^p = U * S^p * V^T
            u, s, v = torch.svd(matrix_f32)
            
            # [Stability] Clamp Singular Values
            # Very small eigenvalues cause huge updates when raised to negative power (s^-0.25)
            s = torch.clamp(s, min=1e-5) 
            
            # Reconstruct
            return (u @ torch.diag(s.pow(power)) @ v.t()).to(matrix.dtype)
        except:
            # Fallback identity if SVD fails (prevents crash)
            return torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)