import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gbm_configs.cond_ref_net.fmnist import get_config_sampling as get_config_sampling_cond_fmnist
from gbm_configs.cond_ref_net.mnist import get_config_sampling as get_config_sampling_cond_mnist
from gbm_configs.cond_ref_net.kmnist import get_config_sampling as get_config_sampling_cond_kmnist
from gbm_configs.NCSNDEEPER.fmnist import get_config_sampling as get_config_sampling_ncsndeeper_fmnist
from gbm_configs.NCSNDEEPER.mnist import get_config_sampling as get_config_sampling_ncsndeeper_mnist
from gbm_configs.NCSNDEEPER.kmnist import get_config_sampling as get_config_sampling_ncsndeeper_kmnist
from gbm_configs.NCSNV3.cifar10 import get_config_sampling as get_config_sampling_cifar10
from calculate_FID import calculate_fid_kid_tensor

from sampling import run_sampler,generate_samples
import numpy as np
import torch
from functools import partial
from torchvision.utils import save_image
import csv
import math
from tqdm import tqdm



if __name__ == "__main__":
    rank =local_rank= int(os.environ['LOCAL_RANK'])
    world_size = 2
    torch.distributed.init_process_group(backend='nccl',init_method='env://')
    torch.cuda.set_device(local_rank)
    device = f'cuda:{rank}'
    show=False
    config = get_config_sampling_cifar10(path ="model_weights/emamodels/cifar10/ema_step50000.pth",device=device)
    config.rank = rank
    model = config.model_class(config=config.model_config()).to(device)

    sd = torch.load(config.path,map_location=config.device)
    N=1000

    delta = 0.00011421052631578947
    sigmas = 0.8 * torch.ones(N).to(config.device)
    mu = 0.5 * sigmas ** 2
    sigma_t = sigmas[0]
    mu_t = mu[0]
    save_path = 'final_samples_tweedie4'
    os.makedirs(save_path,exist_ok=True)


    with torch.no_grad():
        for current_step in range(13):
            x_now = torch.load(f'final_samples3/class_averaged_lamperti_sampling_for_cifar10/25000/cifar10_0.9995/Final_samples_{current_step}_{rank}.pt',map_location=device).clone()
            batch_size = x_now.shape[0]
            print(batch_size)
            for step in tqdm(range(25), desc='Tweedie') if config.show_progress else range(25):
                current_t = torch.full((batch_size,), 0,
                                    dtype=torch.long).to(device, dtype=torch.long)
                
                score = model(x_now.float(),current_t)
                x_now = x_now * torch.exp(delta * (sigma_t ** 2) * (x_now * score) - delta * (mu_t - 1.5 * (sigma_t**2)) * torch.ones_like(x_now))
                temp= torch.clamp(x_now,1.0,2.0) - 1.0
                torch.save(temp,f"{save_path}/tweedie_score_{current_step}_{config.rank}_{step}.pt")

