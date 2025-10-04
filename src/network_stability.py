import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from archs import load_architecture
from utilities import get_loss_and_acc, compute_losses, iterate_dataset
from data import load_dataset

def h_crit(z, rho):
    """Compute H_crit(z) based on the given equation."""
    sqrt_term = np.sqrt(z / 3)
    arcsinh_term = np.arcsinh((3 / rho) * (1 / sqrt_term))
    return 2 * sqrt_term * np.sinh((1 / 3) * arcsinh_term)

def run_stability_analysis(lr, h0, post_samples, dataset, arch_id, loss, physical_batch_size, device_id):
    # Load device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # Load dataset and model
    train_dataset, _ = load_dataset(dataset, loss, device)
    loss_fn, acc_fn = get_loss_and_acc(loss)
    #model_path = "RESULTS/cifar10-10k/fc-tanh/seed_0/mse/gd/lr_0.01_wd_0.0/snapshot_final"
    model_path = "RESULTS/cifar10-10k/resnet20/seed_0/mse/gd/lr_0.05_wd_0.0/snapshot_final"
    model = load_architecture(arch_id, dataset).to(device)
    model.load_state_dict(torch.load(model_path))

    # Get initial parameters and compute initial loss
    initial_params = parameters_to_vector(model.parameters()).detach()
    initial_loss, _ = compute_losses(model, [loss_fn, acc_fn], train_dataset, physical_batch_size, device)
    print("initial loss is",initial_loss)
    # Compute gradient norm at the initial point
    model.zero_grad()
    for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
        X, y = X.to(device), y.to(device)
        loss = loss_fn(model(X), y) / len(train_dataset)
        loss.backward()
    initial_grad = parameters_to_vector([p.grad for p in model.parameters()])
    initial_grad_norm = torch.norm(initial_grad) ** 2
    print("grad is", initial_grad_norm)

    # Compute z
    num_params = initial_params.numel()
    trace_covariance_noise = h0 * num_params
    print("covariance noise is",trace_covariance_noise)
    z = (post_samples * initial_grad_norm) / trace_covariance_noise

    num_trials = 10
    descent_count = 0
    for trial in range(num_trials):
        avg_grad = torch.zeros_like(initial_params, device=device)

        for _ in range(post_samples):
            noise = torch.normal(0, h0**0.5, size=initial_params.shape, device=device)
            noisy_params = initial_params + noise
            vector_to_parameters(noisy_params, model.parameters())
            model.zero_grad()
            for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
                X, y = X.to(device), y.to(device)
                loss = loss_fn(model(X), y) / len(train_dataset)
                loss.backward()
            noisy_grad = parameters_to_vector([p.grad for p in model.parameters()])
            avg_grad += noisy_grad

        avg_grad /= post_samples
        trial_params = initial_params - lr * avg_grad
        vector_to_parameters(trial_params, model.parameters())
        next_loss, _ = compute_losses(model, [loss_fn, acc_fn], train_dataset, physical_batch_size, device)
        print("next_loss is",next_loss)
        if next_loss < initial_loss:
            descent_count += 1

    descent_probability = descent_count / num_trials
    return z.cpu().numpy(), descent_probability

def main():
    # Grid for lr and h0
    lr_values = np.linspace(0.04, 0.06, 10)  # Linear grid for lr
    inv_values = np.linspace(1e20, 1e29, 10)  # Linear grid for 1/h0
    h0_values = 1 / inv_values  # Compute h0

    # Matrix to store descent probabilities
    probability_matrix = np.zeros((len(h0_values), len(lr_values)))
    z_values = np.zeros(len(h0_values))

    # Run experiments
    dataset = "cifar10-10k"
    arch_id = "resnet20"
    loss = "mse"
    post_samples = 5
    physical_batch_size = 1000
    device_id = 4

    for i, h0 in enumerate(h0_values):
        z_value_row = []
        for j, lr in enumerate(lr_values):
            print(f"Running experiment with lr={lr}, h0={h0}")
            z, descent_probability = run_stability_analysis(
                lr=lr,
                h0=h0,
                post_samples=post_samples,
                dataset=dataset,
                arch_id=arch_id,
                loss=loss,
                physical_batch_size=physical_batch_size,
                device_id=device_id,
            )
            z_value_row.append(z)
            probability_matrix[i, j] = descent_probability
            hcrit_value = h_crit(z, rho=lr)  # Use lr as rho in h_crit
            print(f"Result: lr={lr}, z={z}, Descent Probability={descent_probability:.2f}, H_crit={hcrit_value:.6f}")
        z_values[i] = np.mean(z_value_row)

    # Plot results
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=0, vmax=1)

    plt.pcolormesh(lr_values, z_values, probability_matrix, cmap=cmap, norm=norm, shading="auto")
    plt.xscale("linear")
    plt.yscale("linear")
    plt.xlabel("Learning Rate (lr)", fontsize=18)
    plt.ylabel(r"$z$", fontsize=18)
    plt.title("Descent Probability", fontsize=20)

    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca())
    cbar.set_label("Descent Probability", fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    plt.savefig("descent_probability_resnet20_z_vs_lr.png", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
