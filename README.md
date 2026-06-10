# Dale Meets Langevin A Multiplicative Denoising Diffusion Model


*This repository is extended from the [NCSNv3 repository by Song et al.](https://github.com/yang-song/score_sde_pytorch)*


## Project Structure

```text
mninimal_GBM/
│
├── main.py                     # Central entrypoint for training and sampling runs
├── losses.py                   # GBM and score-matching loss definitions
├── utils.py                    # Dataset loading (K-MNIST, etc.), config parsing, EMA utilities
├── sde_lib.py                  # Stochastic Differential Equation (SDE) definitions
├── cond_refinenet_dilated.py   # Architecture definition for Conditional Dilated RefineNet
├── environment.yml             # Conda environment configuration
│
├── configs/                    # Baseline and generic configurations
│   ├── subvp/
│   ├── ve/
│   └── vp/
│
├── gbm_configs/                # Model and Training Configuration (Python)
│   ├── cond_ref_net/           # Configs for Conditional RefineNet (MNIST, F-MNIST, K-MNIST)
│   ├── NCSNDEEPER/             # Configs for NCSNDeeper models
│   └── NCSNV3/                 # Configs for NCSN++ models (CIFAR10, CelebA)
│
├── yamal_configs/              # YAML-based hyperparameter files matched to the python configs
│   ├── cond_ref_net/
│   └── NCSNDEEPER/
│
├── models/                     # Additional Model Architectures and Blocks
│   ├── ncsndeeper/
│   ├── ncsnv2_models/
│   ├── ddpm.py
│   ├── ncsnpp.py
│   └── ncsnv2.py
│
└── mnist_exps/                 # Execution logic 
    ├── calculate_FID.py        # FID metric evaluation script
    ├── cond_refinenet_dilated.py
    ├── nearest_neighbours.py
    ├── sample_search.py
    ├── sampling.py             # Modular sampling logic (dls vs ums)
    ├── train_end_to_end.py     # Training loop orchestrator
    └── tweedie.py
```

## Architectures Supported
- **CondRefineNetDilated:** A conditional refinement network built with multi-resolution dilated convolutions, `ConditionalInstanceNorm2dPlus`, and Conditional Residual Blocks.
- **NCSNv3 (NCSN++):** The standard workhorse architecture from Score-Based Generative Modeling.
- **NCSNDeeper:** An extended depth implementation of the NCSN architecture.

## Usage

### Installation

To run this repository, you must create and install the conda environment from the provided `environment.yml` file:
```bash
conda env create -f environment.yml
```
After installation, activate the environment before running any scripts:
```bash
conda activate <env_name>
```
Alternatively, you can prepend `conda run -n <env_name>` to your commands.

The repository uses a centralized CLI within `main.py` which dynamically loads the correct data and model configuration based on the arguments provided.

### Training a model
Run the `main.py` script with the `--mode train` flag. Multi-GPU training is supported via PyTorch Elastic Launch.
```bash
python main.py --mode train --model ncsn_deeper --dataset fmnist --num_gpus 1
```

### Sampling from a model
Generate images from a trained model using the requested SDE solver:
```bash
python main.py --mode sample --model ncsn_deeper --dataset mnist --sampler_mode dls --model_path path/to/your_model.pth
```

### Supported CLI Arguments
The `main.py` script arguments are logically grouped as follows. You can view them at any time by running `conda run -n <env_name> python main.py --help`.

**Core Configuration:**
- `--mode`: Execution mode (`train`, `sample`, `calculate_fid_kid`, `calculate_fid`, or `nearest_neighbors`).
- `--model`: Model architecture to use (`cond_ref_net`, `ncsn_deeper`, `ncsnv3`).
- `--dataset`: Dataset to train or sample from (`mnist`, `fmnist`, `kmnist`, `cifar10`).
- `--model_path`: Path to an existing model checkpoint for sampling or resume.

**Sampling Configuration:**
- `--sampler_mode`: sampler mode to use (`dls` or `ums`). Default is `dls`.
- `--sampler_batch_size`: Batch size used during sampling. Default is `2048`.
- `--num_of_samples`: Total number of samples to generate. Default is `50000`.
- `--noise_start_point`: Initialization point for the diffusion sampling process (`class_averaged` or `noise`). Default is `noise`.
- `--display_tweedie_images_intermediate`: Visually display intermediate Tweedie generated images during sampling.

**I/O Configuration:**
- `--save_path`: Directory path to save generated images or outputs.
- `--save_samples`: If set, generated samples will be saved to disk.
- `--save_intermediate_tensor`: Path to save the intermediate trajectory tensors.

**Distributed Training Configuration:**
- `--num_gpus`: Number of GPUs to use. `>1` automatically invokes PyTorch distributed data parallel (DDP). Default is `1`.
- `--port`: Master port for DDP distributed training. Default is `29500`.

*Note: Models and datasets must have a matching YAML and config file, e.g., using `cifar10` works with `ncsnv3`.*

## Configuration Details
The project utilizes a tiered configuration system where the original architecture configurations are defined in YAML, and our custom GBM variables are extended on top via Python scripts.

1. **Python Configs (`gbm_configs/`)**: These configs (available for `NCSNDEEPER`, `CONDREF`, and `NCSNV3`) extend the base model parameters with GBM-specific hyperparameters. They define PyTorch `transforms`, target `batch_size`, learning rates, and time-dependent noise schedules (`mu`, `sigma`).
   - UI toggles like `config.interactive = True`, `config.show_progress = True`, and `config.do_tweedie = True` (Tweedie's formula correction during sampling) live here.
2. **YAML Configs (`yamal_configs/`)**: These are the original structural parameter files for the baseline models (e.g., detailing `num_scales`, `channels`, `ngf` sizes).
