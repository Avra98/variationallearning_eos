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

#### Variational Learning Experiments
```bash
# Run VL experiments
python src/gd.py --method ivon --lr 0.1 --mc_samples 10
```

#### Quadratic Analysis
```bash
# Run quadratic dynamics analysis
python quad_dyn_sigma_n.py
```

### Key Scripts

- `src/gd.py`: Main training script with VL and standard GD
- `src/network_stability.py`: Network stability analysis
- `quad_dyn_sigma_n.py`: Quadratic dynamics analysis
- `src/utilities.py`: Utility functions for analysis

## Experiments

### 1. Edge of Stability Analysis
- Analysis of sharpness dynamics during training
- Comparison between standard GD and VL
- Network-specific critical sharpness values

### 2. Variational Learning Dynamics
- Posterior covariance analysis
- Monte Carlo sample effects
- Flatter solution finding

### 3. Quadratic Problem Analysis
- Theoretical foundations
- Dynamics of VL on quadratic objectives
- Connection to EoS framework

## Key Findings

1. **Flatter Solutions**: VL finds flatter solutions compared to standard GD
2. **Posterior Covariance Control**: Controlling posterior covariance affects solution flatness
3. **Monte Carlo Samples**: Number of MC samples influences the flatness of found solutions
4. **Network Generalization**: Results hold across different architectures (ResNet, ViT, FC networks)

## Reproducing Results

### Paper Figures

The main paper figures can be reproduced by running:

```bash
# Figure 1: EoS dynamics comparison
python src/network_stability.py --generate_figures

# Figure 2: VL sharpness analysis
python src/gd.py --analysis sharpness --method ivon

# Figure 3: Quadratic dynamics
python quad_dyn_sigma_n.py --plot_dynamics
```

### Experiment Configurations

All experiment configurations are documented in the respective script files. Key parameters:

- `--lr`: Learning rate
- `--mc_samples`: Number of Monte Carlo samples for VL
- `--arch`: Network architecture
- `--dataset`: Dataset to use

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ghosh2024variational,
  title={Variational Learning Finds Flatter Solutions at the Edge of Stability},
  author={Ghosh, Avrajit and Cong, Bai and Yokota, Rio and Ravishankar, Saiprasad and Wang, Rongrong and Tao, Molei and Khan, Mohammad Emtiyaz and Möllenhoff, Thomas},
  journal={arXiv preprint arXiv:2506.12903},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions about this repository, please contact:
- Avrajit Ghosh: [GitHub](https://github.com/Avra98)
- Paper: [arXiv:2506.12903](https://arxiv.org/abs/2506.12903)

## Acknowledgments

We gratefully acknowledge support from the Simons Foundation and all contributors to this work.