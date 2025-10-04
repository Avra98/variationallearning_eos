import argparse
import math
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
from torch.optim import SGD, Adam
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST
from transformers import AutoConfig, AutoModelForImageClassification

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ivon._ivon import IVON
from resnet_cifar import resnet20
from utilities_vit_resnet import (get_hessian_eigenvalues,
                                  get_preconditioned_hessian_eigenvalues)


def evaluate(network, data, loss_fn, target, batch_size):
    network.eval()
    with torch.inference_mode():
        chunk_num = math.ceil(len(data) / batch_size)
        loss = 0
        correct = 0
        for (X, y) in zip(data.chunk(chunk_num), target.chunk(chunk_num)):
            preds = network(X)
            loss += loss_fn(preds, y) / len(data)
            if target.dim() == 1:   # CE
                correct += (preds.argmax(dim=1) == y).sum()
            elif target.dim() == 2:  # MSE
                correct += (preds.argmax(dim=1) == y.argmax(dim=1)).sum()
    return loss.item(), correct.item() / len(data)


def main(args):

    print(args)

    # Initialization
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Workaround for HF transformers
    class WrapperModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x).logits

    if args.dataset.startswith("cifar"):
        num_channels = 3
        if args.dataset == "cifar10":
            num_classes = 10
            train_dataset = CIFAR10(root="datasets", train=True)
            test_dataset = CIFAR10(root="datasets", train=False)
        elif args.dataset == "cifar100":
            num_classes = 100
            train_dataset = CIFAR100(root="datasets", train=True)
            test_dataset = CIFAR100(root="datasets", train=False)
        train_data = torch.tensor(train_dataset.data.transpose(0, 3, 1, 2))
        train_label = torch.tensor(train_dataset.targets)
        test_data = torch.tensor(test_dataset.data.transpose(0, 3, 1, 2))
        test_label = torch.tensor(test_dataset.targets)

    elif args.dataset == "fashionmnist":
        num_channels = 1
        num_classes = 10
        train_dataset = FashionMNIST(root="datasets", train=True)
        test_dataset = FashionMNIST(root="datasets", train=False)
        train_data = torch.zeros(len(train_dataset), 1, 32, 32, dtype=torch.uint8)
        test_data = torch.zeros(len(test_dataset), 1, 32, 32, dtype=torch.uint8)
        train_data[:, :, 2:30, 2:30] = train_dataset.data[:, None]
        train_label = train_dataset.targets
        test_data[:, :, 2:30, 2:30] = test_dataset.data[:, None]
        test_label = test_dataset.targets

    if args.subset_size > 0:
        num_samples = args.subset_size // num_classes
        idx_list = []
        for i in range(num_classes):
            idx_list.extend(list(torch.where(train_label == i)[0][:num_samples]))
        perm = torch.randperm(len(idx_list))
        train_data = train_data[idx_list][perm]
        train_label = train_label[idx_list][perm]

    train_data, train_label = train_data.float().to(device), train_label.to(device)
    test_data, test_label = test_data.float().to(device), test_label.to(device)
    mean = train_data.mean(dim=(0, 2, 3), keepdim=True)
    std = train_data.std(dim=(0, 2, 3), keepdim=True)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    del mean, std

    if args.loss == "mse":
        train_label = torch.nn.functional.one_hot(train_label, num_classes=num_classes)
        test_label = torch.nn.functional.one_hot(test_label, num_classes=num_classes)

    if args.loss == "mse":
        class SquaredLoss(torch.nn.Module):
            def forward(self, input: torch.Tensor, target: torch.Tensor):
                return 0.5 * ((input - target) ** 2).sum()
        loss_fn = SquaredLoss()
    elif args.loss == "ce":
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    if args.arch == "vit":
        config = AutoConfig.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        config.num_labels = num_classes
        config.image_size = 32
        config.patch_size = 8
        config.num_hidden_layers = 2
        config.num_channels = num_channels
        network = AutoModelForImageClassification.from_config(config, attn_implementation="eager")
        network = WrapperModel(network)
    elif args.arch == "resnet20":
        network = resnet20(num_classes=num_classes, input_channels=num_channels)
    network.to(device)

    if args.opt == "gd":
        optimizer = SGD(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "adam":
        optimizer = Adam(
            network.parameters(), lr=args.lr, betas=(0.0, 0.999), weight_decay=args.weight_decay
        )
    elif args.opt == "ivon":
        optimizer = IVON(
            network.parameters(), lr=args.lr, ess=args.ess,
            hess_init=args.hess_init, beta1=0, beta2=args.ivon_beta2,
            weight_decay=args.weight_decay, rescale_lr=(args.ivon_beta2 == 1), alpha=args.alpha
        )

    for step in range(args.max_steps):

        if args.eig_freq != -1 and step % args.eig_freq == 0 and step > 0:
            eig = get_hessian_eigenvalues(
                network, loss_fn, train_data, train_label,
                neigs=args.neigs, physical_batch_size=args.physical_batch_size, device=device
            )
            print("eigenvalues: ", eig, flush=True)

            if args.opt == "adam" or (args.opt =="ivon" and args.ivon_beta2 != 1):
                pre_eig = get_preconditioned_hessian_eigenvalues(
                    network, loss_fn,train_data, train_label, optimizer,
                    neigs=args.neigs, physical_batch_size=args.physical_batch_size, device=device
                )
                print("precond-eigs: ", pre_eig)

            if args.opt == "ivon":
                hess = optimizer.state_dict()["param_groups"][0]["hess"]
                print(hess.max(), hess.min(), hess.mean())

        if step % 10 == 0:
            train_loss, train_acc = evaluate(
                network, train_data, loss_fn, train_label, args.physical_batch_size
            )
            test_loss, test_acc = evaluate(
                network, test_data, loss_fn, test_label, args.physical_batch_size
            )
            print(f"{step}\t{train_loss:.3f}\t{train_acc:.3f}\t{test_loss:.3f}\t{test_acc:.3f}")

        loss_sum = 0
        chunk_num = math.ceil(len(train_data) / args.physical_batch_size)
        network.train()
        for _ in range(args.post_samples):
            with optimizer.sampled_params(train=True) if optimizer.__class__.__name__ == "IVON" else nullcontext():
                for (X, y) in zip(train_data.chunk(chunk_num), train_label.chunk(chunk_num)):
                    output = network(X)
                    loss = loss_fn(output, y) / len(train_data)
                    loss_sum += loss
                    loss.backward()
        # print(f"Step {step}, loss: {loss_sum:.3f}")
        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using gradient descent.")

    parser.add_argument("--arch", type=str, default="vit", choices=["vit", "resnet20"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "fashionmnist"])
    parser.add_argument("--subset_size", type=int, default=-1)
    parser.add_argument("--loss", type=str, default="mse", choices=["ce", "mse"])
    parser.add_argument("--max_steps", type=int, default=10000)

    parser.add_argument("--opt", type=str, default="gd", choices=["gd", "ivon", "adam"])
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--ess", type=int, default=5e4)
    parser.add_argument("--hess_init", type=float, default=1)
    parser.add_argument("--ivon_beta2", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument("--post_samples", type=int, default=1)

    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--physical_batch_size", type=int, default=10000, help="Max examples per GPU batch")
    parser.add_argument("--neigs", type=int, default=2, help="Number of top eigenvalues to compute")
    parser.add_argument("--eig_freq", type=int, default=-1, help="Frequency for computing Hessian eigenvalues")
    args = parser.parse_args()

    main(args)
