import math
import os
import sys

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import Dataset, DataLoader
from utilities import CustomAdam, compute_preconditioned_hvp, get_preconditioned_hessian_eigenvalues, iterate_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utilities import DEFAULT_PHYS_BS, get_loss_and_acc


def get_param_with_grad(net):
    """Return the parameters of the network that require gradients."""
    return [param for param in net.parameters() if param.requires_grad]


def compute_hvp_vit(
    network, loss_fn, data, target, vector, device, trainable_only, physical_batch_size
):
    """Compute a Hessian-vector product for ViTs, considering only trainable parameters."""
    n = len(data)
    hvp = torch.zeros_like(vector, device=device)
    chunk_num = math.ceil(len(data) / physical_batch_size)
    for X, y in zip(data.chunk(chunk_num), target.chunk(chunk_num)):
        loss = loss_fn(network(X), y) / n
        grads = torch.autograd.grad(
            loss,
            inputs=get_param_with_grad(network) if trainable_only else network.parameters(),
            create_graph=True,
            allow_unused=True,
        )
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [
            g.contiguous()
            for g in torch.autograd.grad(
                dot,
                get_param_with_grad(network) if trainable_only else network.parameters(),
                retain_graph=True,
            )
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
    network, loss_fn, data, target, trainable_only, neigs, physical_batch_size=1000, device="cuda"
):
    """Compute the leading Hessian eigenvalues for ViTs."""
    hvp_delta = lambda delta: compute_hvp_vit(
        network, loss_fn, data, target, delta, device, trainable_only, physical_batch_size
    ).detach()
    params = get_param_with_grad(network) if trainable_only else network.parameters()
    nparams = len(parameters_to_vector(params).to(device))
    evals, evecs = lanczos(hvp_delta, nparams, neigs, device)
    return evals


def compute_preconditioned_hvp_vit(
    network, loss_fn, data, target, vector, device, trainable_only, optimizer, physical_batch_size=1000
):
    """Compute the preconditioned Hessian-vector product for ViTs, similar to compute_hvp_vit."""
    n = len(data)
    hvp = torch.zeros_like(vector, device=device)
    chunk_num = math.ceil(len(data) / physical_batch_size)
    for X, y in zip(data.chunk(chunk_num), target.chunk(chunk_num)):
        X, y = X.to(device), y.to(device)
        loss = loss_fn(network(X), y) / n
        grads = torch.autograd.grad(
            loss,
            inputs=get_param_with_grad(network) if trainable_only else network.parameters(),
            create_graph=True,
            allow_unused=True,
        )
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [
            g.contiguous()
            for g in torch.autograd.grad(
                dot,
                get_param_with_grad(network) if trainable_only else network.parameters(),
                retain_graph=True,
            )
        ]
        hvp += parameters_to_vector(grads)

    # Apply preconditioning
    if type(optimizer).__name__ == "CustomAdam":
        preconditioner = get_adam_preconditioner(optimizer, device)
    else:
        preconditioner = get_ivon_preconditioner(optimizer, device)

    hvp_preconditioned = hvp * preconditioner
    return hvp_preconditioned



def get_preconditioned_hessian_eigenvalues_vit(
    network, loss_fn, data, target, trainable_only,optimizer, neigs=6, physical_batch_size=1000, device="cuda"
):
    """Compute the leading preconditioned Hessian eigenvalues for ViTs."""
    hvp_delta = lambda delta: compute_preconditioned_hvp_vit(
        network, loss_fn, data, target, delta, device, trainable_only, optimizer, physical_batch_size
    ).detach()
    params = get_param_with_grad(network)
    nparams = len(parameters_to_vector(params).to(device))
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


    

def calculate_hcrit_projected(network, loss_fn, dataset, lr, post_samples, h0, ess, physical_batch_size, device):
    """
    Calculate the critical Hessian threshold (H_crit) for a given model state.

    Args:
        network: PyTorch neural network model.
        loss_fn: Loss function used.
        dataset: Dataset object for computing gradients.
        lr: Learning rate.
        post_samples: Number of posterior samples (Ns).
        h0: Posterior variance initialization (diagonal elements of \Sigma).
        ess: Effective sample size.
        physical_batch_size: Batch size for data iteration.
        device: Device to perform computation ('cuda' or 'cpu').

    Returns:
        Hcrit: The critical Hessian threshold value.
    """
    projection,_ = project_gradient_top_eigenvector(network, loss_fn, dataset, physical_batch_size, device)
    print("projection is",projection)
    var = (1 / (ess * h0))
    # Calculate z
    z = (post_samples * (projection**2)) / var
    if z <= 0:
        raise ValueError(f"Invalid z value: {z}. Ensure z > 0 for Hcrit computation.")

    # Compute H_crit
    arg = torch.tensor(((3 / lr)) * (3 / z) ** 0.5, device=device)
    if not torch.isfinite(arg).item():
        raise ValueError(f"Invalid argument to asinh: {arg}. Ensure proper input values.")
    Hcrit = 2 * (z / 3) ** 0.5 * torch.sinh((1 / 3) * torch.asinh(arg)).item()

    return Hcrit



    
def project_gradient_top_eigenvector(network, loss_fn, data, target,trainable only,neigs, physical_batch_size, device,neigs) -> float:
    """
    Calculate the projection of the gradient vector onto the top eigenvector of the Hessian.

    Args:
        network: PyTorch neural network model.
        loss_fn: Loss function used.
        dataset: Dataset for computing Hessian and gradient.
        physical_batch_size: Batch size for dataset iteration.
        neigs: Number of leading eigenvalues/eigenvectors to compute (default is 1).
        device: Device to perform computation ('cuda' or 'cpu').

    Returns:
        Projection value: v1^T g, where v1 is the top eigenvector of the Hessian, and g is the gradient.
    """
    
    # Compute the gradient vector g
    g = compute_gradient(network, loss_fn, data,target, physical_batch_size, device)

    # Print the norm of the gradient
    gradient_norm = g.norm().item()

    # Compute the top eigenvector of the Hessian
    evals, evecs = get_hessian_eigenvalues(network, loss_fn, data, target,trainable_only, neigs=neigs,
                                           physical_batch_size=, device=)

    # Extract the top eigenvector (corresponding to the largest eigenvalue)
    v1 = evecs[:, 0]

    # Compute the projection v1^T g (dot product between v1 and g)
    projection = torch.dot(v1, g).item()
    align = projection/gradient_norm
    print("alignment ratio",align)
    # print("projetion and grad norm are",projection,gradient_norm)
    return projection,align




def compute_gradient(network, loss_fn, dataset, physical_batch_size,device):
    """ Compute the gradient of the loss function at the current network parameters. """
    
    # Move network to the specified device
    network.to(device)
    
    # Initialize gradient on the specified device
    p = len(parameters_to_vector(network.parameters()))
    average_gradient = torch.zeros(p, device=device)

    # Iterate through the dataset and compute gradients
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        X, y = X.to(device), y.to(device)
        #print(X.shape,y.shape)
        batch_loss = loss_fn(network(X), y) / len(dataset)
        batch_gradient = parameters_to_vector(torch.autograd.grad(
            batch_loss, inputs=network.parameters()))
        average_gradient += batch_gradient

    return average_gradient

