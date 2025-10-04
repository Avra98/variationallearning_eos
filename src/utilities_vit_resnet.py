import math

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh
from torch.nn.utils import parameters_to_vector


def compute_hvp(
    network, loss_fn, data, target, vector, device, physical_batch_size
):
    """Compute a Hessian-vector product."""
    n = len(data)
    hvp = torch.zeros_like(vector, device=device)
    chunk_num = math.ceil(len(data) / physical_batch_size)
    for X, y in zip(data.chunk(chunk_num), target.chunk(chunk_num)):
        loss = loss_fn(network(X), y) / n
        grads = torch.autograd.grad(
            loss, inputs=network.parameters(), create_graph=True, allow_unused=True
        )
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [
            g.contiguous()
            for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)
        ]
        hvp += parameters_to_vector(grads)
    return hvp


def lanczos(matrix_vector, dim: int, neigs: int, device: str):
    """Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors."""

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float, device=device)
        result = matrix_vector(gpu_vec)
        return result.cpu().numpy()

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return (
        torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).to(device).float(),
        torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).to(device).float(),
    )


def get_hessian_eigenvalues(
    network, loss_fn, data, target, neigs, physical_batch_size=1000, device="cuda"
):
    """Compute the leading Hessian eigenvalues."""
    def hvp_delta(delta):
        return compute_hvp(
            network, loss_fn, data, target, delta, device, physical_batch_size
        ).detach()
    nparams = sum(p.numel() for p in network.parameters())
    evals, evecs = lanczos(hvp_delta, nparams, neigs, device)
    return evals


def compute_preconditioned_hvp(
    network, loss_fn, data, target, vector, device, optimizer, physical_batch_size=1000
):
    """Compute the preconditioned Hessian-vector product, similar to compute_hvp."""

    hvp = compute_hvp(network, loss_fn, data, target, vector, device, physical_batch_size)

    # Apply preconditioning
    if type(optimizer).__name__ == "Adam":
        preconditioner = get_adam_preconditioner(optimizer, device)
    else:
        preconditioner = get_ivon_preconditioner(optimizer, device)

    return hvp * preconditioner


def get_preconditioned_hessian_eigenvalues(
    network, loss_fn, data, target, optimizer, neigs=6, physical_batch_size=1000, device="cuda"
):
    """Compute the leading preconditioned Hessian eigenvalues."""
    def hvp_delta(delta):
        return compute_preconditioned_hvp(
            network, loss_fn, data, target, delta, device, optimizer, physical_batch_size
        ).detach()
    nparams = sum(p.numel() for p in network.parameters())
    evals, evecs = lanczos(hvp_delta, nparams, neigs, device)
    return evals


def get_adam_preconditioner(optimizer, device):
    preconditioner = []
    for group in optimizer.param_groups:
        #print(group)
        for p in group['params']:
            #print("in loop",p.grad)
            if p.grad is None:
                continue
            state = optimizer.state[p]
            #print("state is", state)
            if 'exp_avg_sq' in state:
                v_t = state['exp_avg_sq']
                step = state['step']
                beta2 = group['betas'][1]
                beta1 = group['betas'][0]
                # Bias-corrected second moment
                v_t_hat = v_t / (1 - beta2 ** step)
                # Inverse preconditioner (element-wise reciprocal of the square root of v_t_hat)
                inv_preconditioner = (1 - beta1 ** step) / (torch.sqrt(v_t_hat) + group['eps'])
                preconditioner.append(inv_preconditioner.flatten())
    return torch.cat(preconditioner).to(device)

def get_ivon_preconditioner(ivon_optimizer, device):
    preconditioner = []
    for group in ivon_optimizer.param_groups:
        hess = group['hess']  # Extract the Hessian approximation once for each group
        step = ivon_optimizer.current_step
        beta1 = group['beta1']

        # Bias-correct the Hessian
        hess_hat = hess

        # Compute the inverse preconditioner using the Hessian
        inv_preconditioner = (1 - beta1 ** step) / (hess_hat + group['weight_decay'])

        # Append the inverse preconditioner only once for each group
        preconditioner.append(inv_preconditioner.flatten())

    # Concatenate all flattened preconditioners into a single tensor and move to the specified device
    return torch.cat(preconditioner).to(device)
