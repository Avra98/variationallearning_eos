from os import makedirs
import torch
import numpy as np

import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ivon._ivon import IVON

from torch.nn.utils import parameters_to_vector
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse

from archs import load_architecture
from utilities import get_gd_optimizer, get_gd_directory, get_loss_and_acc, compute_losses, \
    save_files, save_files_final, get_hessian_eigenvalues, iterate_dataset,  \
    store_singular_values, initialize_subspace_distances, compute_grad_h_grad, second_order_approx, \
    get_preconditioned_hessian_eigenvalues, compute_neg_elbo, calculate_hcrit, get_trace_hessian_cube, get_trace, evaluate_rhs_descent, \
    calculate_hcrit_adjusted,calculate_hcrit_projected, project_gradient_top_eigenvector 
from data import load_dataset, take_first, DATASETS


def main(dataset: str = "cifar10-10k", arch_id: str = "resnet32", loss: str = "ce", opt: str = "gd", lr: float = 1e-2, max_steps: int = 10000, neigs: int = 0,
         physical_batch_size: int = 1000, eig_freq: int = -1, iterate_freq: int = -1, save_freq: int = -1,
         save_model: bool = False, beta: float = 0.0, beta2: float = 0.99, nproj: int = 0,
         loss_goal: float = None, acc_goal: float = None, abridged_size: int = 50000, seed: int = 0, h0: float = 0.1, alpha: int = 10, ess: float = 1e5,
         post_samples: int = 10, ad_noise: float = 1e-6, weight_decay: float = 0.0, device_id: int = 1):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    root_folder = "network_stability"  # Root folder name
    directory = os.path.join(root_folder, get_gd_directory(dataset, lr, arch_id, seed, opt, loss, beta, h0, alpha, post_samples, beta2, ess, ad_noise, weight_decay, physical_batch_size))
    
    print(f"device: {device}")
    print("opt is", opt)
    print(f"output directory: {directory}")
    makedirs(directory, exist_ok=True)


    train_dataset, test_dataset = load_dataset(dataset, loss, device)
    abridged_train = take_first(train_dataset, abridged_size, device)

    loss_fn, acc_fn = get_loss_and_acc(loss)

    torch.manual_seed(seed)
    network = load_architecture(arch_id, dataset).to(device)

    projectors = torch.randn(nproj, len(parameters_to_vector(network.parameters())))

    optimizer = get_gd_optimizer(network.parameters(), opt, lr, beta, post_samples,beta2,h0,alpha,ess,weight_decay)
    print("optimizer is", type(optimizer).__name__)
    #scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=0)

    train_loss, test_loss, train_acc, test_acc = torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps)
    iterates = torch.zeros(max_steps // iterate_freq if iterate_freq > 0 else 0, len(projectors))
    eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    neg_elbo = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0)
    pre_eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    term12_values = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, device=device)
    term3_values = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, device=device)
    H_crit_values = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, device=device)
    Hcrit_adjusted_values = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, device=device)
    align_values = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, device=device)
    for step in range(0, max_steps):
    
        train_loss[step], train_acc[step] = compute_losses(network, [loss_fn, acc_fn], train_dataset,
                                                           physical_batch_size, device)
        test_loss[step], test_acc[step] = compute_losses(network, [loss_fn, acc_fn], test_dataset, physical_batch_size, device)


        if eig_freq != -1 and step % eig_freq == 0 and step !=0:
            # Hcrit_adjusted = calculate_hcrit_projected(network, loss_fn, train_dataset, lr, post_samples, h0, ess, physical_batch_size=physical_batch_size, device=device)
            # print("H_crit adjusted:", Hcrit_adjusted)
            eigs[step // eig_freq, :],_ = get_hessian_eigenvalues(network, loss_fn, abridged_train, neigs=neigs,
                                                                  physical_batch_size=physical_batch_size, device=device )
            #pre_eigs[step // eig_freq, :] = get_preconditioned_hessian_eigenvalues(network, loss_fn, abridged_train, optimizer, neigs=neigs, device=device)
            print("eigenvalues: ", eigs[step//eig_freq, :])
            if opt == "ivon":
                # descent_rhs, term12, term3 = evaluate_rhs_descent(network, loss_fn, train_dataset, lr, post_samples, h0, ess,
                #                                                   physical_batch_size=physical_batch_size, n_iters=50, device=device)
                # term12_values[step // eig_freq] = term12
                # term3_values[step // eig_freq] = term3
                #H_crit = calculate_hcrit(network, loss_fn, train_dataset, lr, post_samples, h0, ess, physical_batch_size=physical_batch_size, device=device)
                Hcrit_adjusted = calculate_hcrit_projected(network, loss_fn, train_dataset, lr, post_samples, h0, ess, physical_batch_size=physical_batch_size, device=device)
                #_, align = project_gradient_top_eigenvector(
                #                    network, loss_fn, train_dataset, physical_batch_size=physical_batch_size, device=device, neigs=1)
                #align = project_gradient_top_eigenvector(network, loss_fn, dataset, physical_batch_size=physical_batch_size, device=device,neigs=1)
                #H_crit_values[step // eig_freq] = H_crit
                Hcrit_adjusted_values[step // eig_freq] = Hcrit_adjusted
                #align_values[step // eig_freq]  = np.abs(align)
                #print("term12 and term3 are", term12, term3)
                print("H_crit adjusted:", Hcrit_adjusted)
                #print("H_crit", H_crit)
                #print("align", np.abs(align))
            

            

        if iterate_freq != -1 and step % iterate_freq == 0:
            iterates[step // iterate_freq, :] = projectors.mv(parameters_to_vector(network.parameters()).cpu().detach())

        if save_freq != -1 and step % save_freq == 0:
            save_files(directory, [("eigs", eigs[:step // eig_freq]),
                                ("iterates", iterates[:step // iterate_freq]),
                                ("train_loss", train_loss[:step]),
                                ("test_loss", test_loss[:step]),
                                ("pre-eigs", pre_eigs[:step // eig_freq]),
                                ("train_acc", train_acc[:step]),
                                ("test_acc", test_acc[:step]),
                                ("term12", term12_values[:step // eig_freq]),
                                ("term3", term3_values[:step // eig_freq]),
                                ("H_crit", H_crit_values[:step // eig_freq]),
                                ("Hcrit_adjusted", Hcrit_adjusted_values[:step // eig_freq]),
                                ("align_values", align_values[:step // eig_freq])])


        if step % 10 == 0:
            print(f"{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t{test_loss[step]:.3f}\t{test_acc[step]:.3f}")

        if (loss_goal != None and train_loss[step] < loss_goal) or (acc_goal != None and train_acc[step] > acc_goal):
            break
        
        if opt=="ivon":
            ## ivon-sgd
            # optimizer.zero_grad()
            # for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
            #     for _ in range(post_samples):
            #         with optimizer.sampled_params(train=True):                   
            #             loss = loss_fn(network(X.to(device)), y.to(device)) / physical_batch_size
            #             loss.backward()
            #     optimizer.step()

            ## ivon-gd
            optimizer.zero_grad()    
            for _ in range(post_samples):
                with optimizer.sampled_params(train=True):   
                    optimizer.zero_grad()
                    for (X, y) in iterate_dataset(train_dataset, physical_batch_size):                      
                        loss = loss_fn(network(X.to(device)), y.to(device)) / len(train_dataset)
                        loss.backward()
            # with torch.no_grad():
            #     avg_grad_noisy = optimizer.get_avg_grad()           
            optimizer.step()



        else:
            optimizer.zero_grad()
            # noise_std = args.ad_noise
            # noise_dict = {}
            for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
                # with torch.no_grad():
                #     for param in network.parameters():
                #         noise=torch.randn_like(param) * noise_std
                #         param.add_(noise)
                #         noise_dict[param]=noise
                        
                loss = loss_fn(network(X.to(device)), y.to(device)) / len(train_dataset)
                #loss = loss_fn(network(X.to(device)), y.to(device)) / physical_batch_size
                loss.backward()

                # with torch.no_grad():
                #     for param in network.parameters():
                #         param.sub_(noise_dict[param])
                
            optimizer.step()

    save_files_final(directory, [("eigs", eigs[:(step + 1) // eig_freq]),
                                ("iterates", iterates[:(step + 1) // iterate_freq]),
                                ("train_loss", train_loss[:step + 1]),
                                ("test_loss", test_loss[:step + 1]),
                                ("train_acc", train_acc[:step + 1]),
                                ("test_acc", test_acc[:step + 1]),
                                ("term12", term12_values[:(step + 1) // eig_freq]),
                                ("term3", term3_values[:(step + 1) // eig_freq]),
                                ("H_crit", H_crit_values[:(step + 1) // eig_freq]),
                                ("Hcrit_adjusted", Hcrit_adjusted_values[:(step + 1) // eig_freq]),
                                ("align_values", align_values[:(step + 1) // eig_freq])])

    if save_model:
        torch.save(network.state_dict(), f"{directory}/snapshot_final")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using gradient descent.")
    parser.add_argument("--dataset", type=str, default="cifar10-10k", choices=DATASETS, help="which dataset to train")
    parser.add_argument("--arch_id", type=str, default="fc-tanh", help="which network architectures to train")
    parser.add_argument("--loss", type=str, default="mse", choices=["ce", "mse"], help="which loss function to use")
    parser.add_argument("--lr", type=float, default=0.05, help="the learning rate")
    parser.add_argument("--max_steps", type=int, default=10000, help="the maximum number of gradient steps to train for")
    parser.add_argument("--opt", type=str, choices=["gd", "polyak", "nesterov", "adam", "ivon"],
                        help="which optimization algorithm to use", default="gd")
    parser.add_argument("--seed", type=int, help="the random seed used when initializing the network weights",
                        default=0)
    parser.add_argument("--beta", type=float, help="momentum parameter (used if opt = polyak or nesterov)", default=0.0)
    parser.add_argument("--beta2", type=float, help="momentum parameter for variance", default=1.0)
    parser.add_argument("--physical_batch_size", type=int,
                        help="the maximum number of examples that we try to fit on the GPU at once", default=1000)
    parser.add_argument("--acc_goal", type=float,
                        help="terminate training if the train accuracy ever crosses this value", default=None)
    parser.add_argument("--loss_goal", type=float, help="terminate training if the train loss ever crosses this value", default=None)
    parser.add_argument("--neigs", type=int, help="the number of top eigenvalues to compute", default=2)
    parser.add_argument("--eig_freq", type=int, default=-1,
                        help="the frequency at which we compute the top Hessian eigenvalues (-1 means never)")
    parser.add_argument("--nproj", type=int, default=0, help="the dimension of random projections")
    parser.add_argument("--iterate_freq", type=int, default=-1,
                        help="the frequency at which we save random projections of the iterates")
    parser.add_argument("--abridged_size", type=int, default=10000,
                        help="when computing top Hessian eigenvalues, use an abridged dataset of this size")
    parser.add_argument("--save_freq", type=int, default=-1,
                        help="the frequency at which we save results")
    parser.add_argument("--save_model", type=bool, default=False,
                        help="if 'true', save model weights at end of training")
    parser.add_argument("--h0", type=float, default=1.0,
                        help="posterior variance init")
    parser.add_argument("--alpha", type=float, default=10.0, help="t-distribution parameter")
    parser.add_argument("--ess", type=float, default=1e5, help="lambda of prior weighting")
    parser.add_argument("--post_samples", type=int, default=10,
                        help="number of posterior samples")
    parser.add_argument("--ad_noise", type=float, default=1e-6,
                        help="ADAM noise injection strength")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="weight decay")
    parser.add_argument("--device_id", type=int, default=1, help="ID of the GPU to use")
    args = parser.parse_args()

    main(dataset=args.dataset, arch_id=args.arch_id, loss=args.loss, opt=args.opt, lr=args.lr, max_steps=args.max_steps,
         neigs=args.neigs, physical_batch_size=args.physical_batch_size, eig_freq=args.eig_freq,
         iterate_freq=args.iterate_freq, save_freq=args.save_freq, save_model=args.save_model, beta=args.beta, beta2=args.beta2,
         nproj=args.nproj, loss_goal=args.loss_goal, acc_goal=args.acc_goal, abridged_size=args.abridged_size,
         seed=args.seed, h0=args.h0, alpha=args.alpha, ess=args.ess, post_samples=args.post_samples, ad_noise=args.ad_noise, weight_decay=args.weight_decay, device_id=args.device_id)