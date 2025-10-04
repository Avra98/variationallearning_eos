import argparse
import math
import os
import sys
from contextlib import nullcontext

import torch
import torchvision
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForImageClassification

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ivon._ivon import IVON
from utilities_vit import get_hessian_eigenvalues, get_loss_and_acc,get_preconditioned_hessian_eigenvalues_vit, calculate_hcrit_projected 
from utilities import CustomAdam, compute_preconditioned_hvp


def evaluate(network, data, loss_fn, target, batch_size):
    network.eval()
    with torch.no_grad():
        chunk_num = math.ceil(len(data) / batch_size)
        loss = 0
        correct = 0
        for (X, y) in zip(data.chunk(chunk_num), target.chunk(chunk_num)):
            preds = network(X)
            loss += loss_fn(preds, y).item() / len(data)
            if target.dim() == 1:   # CE
                correct += (preds.argmax(dim=1) == y).sum().item()
            elif target.dim() == 2:  # MSE
                correct += (preds.argmax(dim=1) == y.argmax(dim=1)).sum().item()
    return loss, correct / len(data)


def main(loss, lr, max_steps, neigs, physical_batch_size, eig_freq,
         seed, weight_decay, device_id, finetune, hessian, opt):

    # Initialization
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Workaround for HF transformers
    class WrapperModel(torch.nn.Module):
        def __init__(self, model, finetune):
            super().__init__()
            self.model = model
            self.finetune = finetune

        def forward(self, x):
            if self.finetune is not None:
                x = torchvision.transforms.functional.resize(x, 224)
            return self.model(x).logits

    # Initialize the network
    if finetune is None:
        config = AutoConfig.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        config.num_labels = 10
        config.image_size = 32
        config.patch_size = 8
        config.num_hidden_layers = 2
        network = AutoModelForImageClassification.from_config(config,attn_implementation="eager")
    else:
        network = AutoModelForImageClassification.from_pretrained(
            "WinKawaks/vit-tiny-patch16-224",
            num_labels=10,
            ignore_mismatched_sizes=True,
            attn_implementation="eager",
        )
        if finetune == "lora":
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["classifier"],
            )
            network = get_peft_model(network, lora_config)
        elif finetune == "lastlayer":
            network.requires_grad_(False)
            network.classifier.requires_grad_(True)

    ft_group = [p for p in network.parameters() if p.requires_grad]

    if hessian == "full":
        network.requires_grad_(True)

    print(f"Total #param: {sum(p.numel() for p in network.parameters())}")
    print(f"Finetune #param: {sum(p.numel() for p in ft_group)}")
    print(f"grad #param: {sum(p.numel() for p in network.parameters() if p.requires_grad)}")

    network = WrapperModel(network, finetune=finetune)
    network = network.to(device)

    # Load CIFAR10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root="datasets", train=True)
    test_dataset = torchvision.datasets.CIFAR10(root="datasets", train=False)

    train_data = (train_dataset.data - 127.5) / 127.5
    train_data = torch.tensor(train_data, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    train_label = torch.tensor(train_dataset.targets, device=device)
    test_data = (test_dataset.data - 127.5) / 127.5
    test_data = torch.tensor(test_data, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    test_label = torch.tensor(test_dataset.targets, device=device)
    if loss == "mse":
        train_label = torch.nn.functional.one_hot(train_label, num_classes=10)
        test_label = torch.nn.functional.one_hot(test_label, num_classes=10)

    loss_fn, acc_fn = get_loss_and_acc(loss)

    if opt == "gd":
        optimizer = torch.optim.SGD(ft_group, lr=lr, weight_decay=weight_decay)
    elif opt == "adam":
        beta1=0.0
        beta2=0.999
        optimizer = CustomAdam(ft_group, lr=lr,betas=(beta1,beta2))
    elif opt == "ivon":
        hess_init = 0.7
        beta2 = 1.0
        ess = 1e6
        print("Hessian init: ", hess_init)
        print("beta2: ", beta2)
        print("ess: ", ess)
        rescale_lr= True if beta2==1 else False
        print("rescale_lr: ", rescale_lr)
        print("lr: ", lr)
        optimizer = IVON(
            ft_group, lr=lr, ess=ess, weight_decay=weight_decay, mc_samples=1,
            beta1=0, beta2=beta2, hess_init=hess_init, rescale_lr=rescale_lr, alpha=10
        )

    eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    hcrit = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    if opt =="adam" or (opt =="ivon" and beta2 !=1):
        pre_eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)


    for step in range(max_steps):

        if eig_freq != -1 and step % eig_freq == 0 and step >0:
            trainable_only = (hessian == "sub")
            eigs[step // eig_freq, :] = get_hessian_eigenvalues(
                network, loss_fn, train_data, train_label, trainable_only,
                neigs=neigs, physical_batch_size=physical_batch_size, device=device
            )
            print("eigenvalues: ", eigs[step // eig_freq, :])   
            hcrit[step // eig_freq, :] = calculate_hcrit_projected(network,loss_fn,train_data)  
            if opt =="adam" or (opt =="ivon" and beta2 !=1):
                pre_eigs[step // eig_freq, :]= get_preconditioned_hessian_eigenvalues_vit(network, loss_fn,train_data, train_label,trainable_only, optimizer, neigs=neigs, physical_batch_size=physical_batch_size, device=device )
                print("precond-eigs: ", pre_eigs[step // eig_freq, :])
        if step % 10 == 0:
            train_loss, train_acc = evaluate(network, train_data, loss_fn, train_label, physical_batch_size)
            test_loss, test_acc = evaluate(network, test_data, loss_fn, test_label, physical_batch_size)
            print(f"{step}\t{train_loss:.3f}\t{train_acc:.3f}\t{test_loss:.3f}\t{test_acc:.3f}", flush=True)

        optimizer.zero_grad()
        loss_sum = 0
        chunk_num = math.ceil(len(train_data) / physical_batch_size)
        network.train()
        with optimizer.sampled_params(train=True) if optimizer.__class__.__name__ == "IVON" else nullcontext():
            for (X, y) in zip(train_data.chunk(chunk_num), train_label.chunk(chunk_num)):
                output = network(X)
                loss = loss_fn(output, y) / len(train_dataset)
                loss_sum += loss
                loss.backward()
        #print(f"Step {step}, loss: {loss_sum:.3f}")
        optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using gradient descent.")
    parser.add_argument("--loss", type=str, default="mse", choices=["ce", "mse"], help="Loss function to use")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum gradient steps to train for")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for initialization")
    parser.add_argument("--physical_batch_size", type=int, default=10000, help="Max examples per GPU batch")
    parser.add_argument("--neigs", type=int, default=2, help="Number of top eigenvalues to compute")
    parser.add_argument("--eig_freq", type=int, default=-1, help="Frequency for computing Hessian eigenvalues")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--device_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--finetune", type=str, default=None, choices=["full", "lora", "lastlayer"], help="Finetuning strategy")
    parser.add_argument("--hessian", type=str, default="full", choices=["full", "sub"], help="Hessian mode")
    parser.add_argument("--opt", type=str, default="gd", choices=["gd", "ivon","adam"], help="Optimizer to use")
    args = parser.parse_args()

    main(
        loss=args.loss, lr=args.lr, max_steps=args.max_steps,
        neigs=args.neigs, physical_batch_size=args.physical_batch_size,
        eig_freq=args.eig_freq, seed=args.seed, weight_decay=args.weight_decay,
        device_id=args.device_id, finetune=args.finetune,
        hessian=args.hessian, opt=args.opt
    )
