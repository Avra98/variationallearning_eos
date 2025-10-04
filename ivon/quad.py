import torch
from torch.autograd import grad

def compute_grad_and_hessian_terms(network, optimizer, loss_fn, X, y, eta, device):
    """
    Computes the values of the following terms:
    1. eta * (true_gradient.T @ noisy_gradient)
    2. (eta^2 / 2) * (noisy_gradient.T @ H @ noisy_gradient)

    Args:
    - network: The model being trained.
    - optimizer: The IVON optimizer.
    - loss_fn: The loss function.
    - X: The input data batch.
    - y: The target data batch.
    - eta: Learning rate.
    - device: The device on which to perform computations.

    Returns:
    - term1: Scalar value of eta * (true_gradient.T @ noisy_gradient)
    - term2: Scalar value of (eta^2 / 2) * (noisy_gradient.T @ H @ noisy_gradient)
    """

    # 1. Compute the true gradient at the mean of the posterior (i.e., current parameters)
    network.zero_grad()
    output = network(X.to(device))
    loss = loss_fn(output, y.to(device))
    true_gradient = grad(loss, network.parameters(), create_graph=True)

    # 2. Sample noisy parameters and compute the noisy gradient
    with optimizer.sampled_params(train=True):
        network.zero_grad()
        output_noisy = network(X.to(device))
        loss_noisy = loss_fn(output_noisy, y.to(device))
        noisy_gradient = grad(loss_noisy, network.parameters(), create_graph=True)

    # 3. Compute the Hessian-vector product for the noisy gradient
    hessian_gradient_product = []
    for g in noisy_gradient:
        hvp = grad(true_gradient, network.parameters(), grad_outputs=g, retain_graph=True)
        hessian_gradient_product.append(hvp)

    # 4. Compute the required terms for the output

    # term1 = eta * (true_gradient.T @ noisy_gradient)
    term1 = sum((tg * ng).sum() for tg, ng in zip(true_gradient, noisy_gradient))
    term1 *= eta

    # term2 = (eta^2 / 2) * (noisy_gradient.T @ H @ noisy_gradient)
    quadratic_form = sum((ng * hvp).sum() for ng, hvp in zip(noisy_gradient, hessian_gradient_product))
    term2 = (eta**2 / 2) * quadratic_form

    return term1.item(), term2.item()

# Example of how to call the function
eta = 0.01  # Example learning rate
term1_value, term2_value = compute_grad_and_hessian_terms(
    network=network, 
    optimizer=optimizer, 
    loss_fn=loss_fn, 
    X=X, 
    y=y, 
    eta=eta, 
    device=device
)

print(f"Term 1 (eta * true_gradient.T @ noisy_gradient): {term1_value}")
print(f"Term 2 ((eta^2 / 2) * noisy_gradient.T @ H @ noisy_gradient): {term2_value}")
