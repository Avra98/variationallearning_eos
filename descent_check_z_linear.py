import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import math
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def quadratic_function(a, x):
    return 0.5 * a * (x - 3) ** 2

def check_loss_change(initial_loss, next_loss):
    if next_loss < initial_loss:
        return 0  # Loss decreased
    elif next_loss > initial_loss:
        return 1  # Loss increased
    else:
        return 2  # Loss stayed the same


def run_experiment(lr, seed, h0, a, post_samples, device_id):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # Set the initial point based on a
    x = torch.tensor([(1 / a) + 3], requires_grad=True, device=device)

    num_trials = 100  # Define the number of trials
    descent_count = 0  # Count how many times descent occurred
    initial_loss = quadratic_function(a, x)

    for trial in range(num_trials):
        avg_grad = torch.zeros_like(x)  # Reset the gradient for each trial

        for _ in range(post_samples):
            # Inject noise into parameters
            noise = torch.normal(mean=0, std=h0**0.5 , size=x.shape, device=device)
            x_noisy = (x + noise).detach().clone().requires_grad_(True)  # Make x_noisy a leaf tensor

            # Compute loss and gradients
            noisy_loss = quadratic_function(a, x_noisy)
            noisy_loss.backward()

            # Accumulate gradients
            avg_grad += x_noisy.grad.clone()
            x_noisy.grad.zero_()

        # Average the gradients
        avg_grad /= post_samples

        # Update parameters using averaged gradients
        with torch.no_grad():
            x_trial = x.clone()  # Clone x for this trial
            x_trial -= lr * avg_grad

        # Compute loss after one step for this trial
        next_loss = quadratic_function(a, x_trial)

        # Check if the loss decreased
        if next_loss.item() < initial_loss.item():
            descent_count += 1  # Increment if descent occurred

    # Compute the probability of descent
    descent_probability = descent_count / num_trials

    print(f"h0: {h0}, post_samples: {post_samples}, a: {a}, Descent Probability: {descent_probability:.2f}")
    return descent_probability


def h_crit(z, rho=1):
    """Compute H_crit(z) based on the given equation."""
    sqrt_term = np.sqrt(z / 3)
    arcsinh_term = np.arcsinh((3 / rho) * (1/sqrt_term))
    return 2 * sqrt_term * np.sinh((1 / 3) * arcsinh_term)



def main(lr, post_samples, seed, device_id):
    inv_values = np.linspace(0.02, 5, 50)  # Linear grid for z (0 to 1 with 50 points)
    h0_values = 1 / inv_values  # Compute h0 from z
    a_values = np.linspace(0.01, 2.5, 50)  # Linear grid for a

    z_values = post_samples * inv_values
    probability_matrix = np.zeros((len(inv_values), len(a_values)))

    for i, h0 in enumerate(h0_values):
        for j, a in enumerate(a_values):
            print(f"Running experiment with h0={h0}, post_samples={post_samples}, a={a}")
            descent_probability = run_experiment(lr, seed, h0, a, post_samples, device_id)
            probability_matrix[i, j] = descent_probability
            print(f"Result: h0={h0}, post_samples={post_samples}, a={a}, Descent Probability={descent_probability:.2f}\n")

    rho = 1  # Set rho value
    h_crit_values = h_crit(z_values, rho)

    # Save variables for plotting
    os.makedirs("descent_figures/saved_variables", exist_ok=True)
    save_path = f"descent_figures/saved_variables/plot_data_lr_{lr:.6f}_post_samples_{post_samples}.npz"
    np.savez(save_path, z_values=z_values, h0_values=h0_values, a_values=a_values,
             probability_matrix=probability_matrix, h_crit_values=h_crit_values, lr=lr, post_samples=post_samples)
    print(f"Saved variables to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-step optimization experiments.")
    parser.add_argument("--lr", type=float, default=1, help="Learning rate for the optimizer.")
    parser.add_argument("--post_samples", type=int, default=10, help="Number of MC samples.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the GPU to use (if available).")

    args = parser.parse_args()

    main(lr=args.lr, post_samples=args.post_samples, seed=args.seed, device_id=args.device_id)


