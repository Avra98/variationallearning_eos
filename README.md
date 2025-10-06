# Variational Learning Finds Flatter Solutions at the Edge of Stability

This repository contains the code and experiments for the paper **"Variational Learning Finds Flatter Solutions at the Edge of Stability"** by Avrajit Ghosh et al.

📄 **Paper**: [arXiv:2506.12903](https://arxiv.org/abs/2506.12903)

## Abstract

Variational Learning (VL) has recently gained popularity for training deep neural networks and is competitive to standard learning methods. Part of its empirical success can be explained by theories such as PAC-Bayes bounds, minimum description length and marginal likelihood, but there are few tools to unravel the implicit regularization in play. Here, we analyze the implicit regularization of VL through the Edge of Stability (EoS) framework. EoS has previously been used to show that gradient descent can find flat solutions and we extend this result to VL to show that it can find even flatter solutions.

## Repository Structure

```
edge-of-stability/
├── src/                          # Core source code
│   ├── archs.py                  # Network architectures
│   ├── gd.py                     # Gradient descent implementations
│   ├── utilities.py              # Utility functions
│   └── network_stability.py      # Network stability analysis
├── data/                         # Data directory (gitignored)
├── figures/                      # Generated figures (gitignored)
├── training_logs/                # Training logs (gitignored)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA (for GPU training)

### Setup

1. Clone the repository:
```bash
git clone git@github.com:Avra98/eos-ivon.git
cd eos-ivon
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

#### Edge of Stability Analysis
```bash
# Run EoS analysis for different networks
python src/network_stability.py --arch fc-tanh --dataset cifar10
python src/network_stability.py --arch resnet20 --dataset cifar10
```

