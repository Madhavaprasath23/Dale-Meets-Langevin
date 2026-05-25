from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from scipy.stats import lognorm
import numpy as np
from tqdm import tqdm
import yaml
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision import datasets, transforms
import math
import torch
from torchvision.transforms import RandomChoice
from torchvision import transforms
import copy
import torch.nn as nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import *
from losses import new_loss




def convert_ddp_to_normal(state_dict):
    parameters = {}
    for name, param in state_dict.items():
        if name.startswith("module."):
            name = name[7:]
        parameters[name] = param
    return parameters


def save_check_point(state_dict, path, save_name):
    os.makedirs(path, exist_ok=True)
    torch.save(state_dict, save_name)


def get_config_path(path):
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
        config = dict2namespace(config)
    return config


def train_model(config):
    if config.multi_gpu_training and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    """ Train an model_class using the provided configuration.

    All arguments are taken from the `config` object.

    Expected fields in `config`:
        - N
        - sigma
        - mu
        - device
        - transforms
        - train_dataset
        - multi_gpu_training
        - num_gpus
        - batch_size
        - config_module
        - continue_model_path
        - continue_model_step
        - do_warmup_continue
        - grad_accum
        - n_iters
        - delta
        - data_set
        - run_name
        - save_path
        - summary_writer_writer
        - optimizer

    The function:
        - Prepares dataset/dataloader (with distributed sampler if multi-GPU).
        - Initializes the NCSN++ model and optimizer.
        - Handles LR warmup, gradient accumulation, and gradient clipping.
        - Performs the training loop with noise perturbations (SDE).
        - Logs loss values and saves checkpoints (best + periodic).
        - Resumes from checkpoints if available.

    Args:
        config: Configuration object containing all required fields.

    Returns:
        None """
    model_class = config.model_class
    config.rank = dist.get_rank() if config.multi_gpu_training else 0
    local_rank = config.rank
    config.device = torch.device(f'cuda:{config.rank}' if torch.cuda.is_available() else 'cpu')
    N = config.N 
    sigmas = config.sigma.to(config.device)
    mu = config.mu.to(config.device)
    transform = config.transforms
    train_dataset = config.train_dataset

    sampler = None if not config.multi_gpu_training else DistributedSampler(
        train_dataset, num_replicas=config.num_gpus, rank=local_rank, shuffle=True)
    shuffle = False if config.multi_gpu_training else True
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=False
    )

    if config.data_set not in ['fmnist','mnist','K-mnist']:
        model_config = config.config_module()
        model = model_class(config=model_config).to(config.rank)
    else:
        config = config.config
        model = model_class(config=config)
    if config.multi_gpu_training:
        model = DDP(model, device_ids=[local_rank])
    else:
        model = model.to(config.device)
    if config.continue_model_path is not None:
            sd = torch.load(config.continue_model_path)['model_state_dict']
            try:
                model.load_state_dict(sd)
            except:
                sd = convert_ddp_to_normal(sd)
                model.load_state_dict(sd)
    opt = config.optim.opt

    learning_rate = config.optim.lr
    warmup_steps = config.optim.warmup if config.optim.warmup> 0 else 0
    grad_clip = config.optim.grad_clip
    weight_decay = config.optim.weight_decay
    beta1 = config.optim.beta1
    eps = config.optim.eps
    ema_step = 5_000 
    optimizer = opt( model.parameters(),lr=learning_rate,
        betas=(beta1,0.999), eps=eps,weight_decay=weight_decay)
    

    if config.continue_model_path is not None:
        try:
            optimizer.load_state_dict(torch.load(config.continue_model_path)['optimizer_state_dict'])
        except:
            pass

    if local_rank == 0 and getattr(config, "summary_writer", None) is None:
        if hasattr(config, "summary_log_dir"):
            from torch.utils.tensorboard import SummaryWriter
            config.summary_writer = SummaryWriter(log_dir=config.summary_log_dir)
    
    grad_accum = config.grad_accum  # change this to higher step to do gradient accumlation
    step = config.continue_model_step if config.continue_model_path else 0
    num_iterations = config.n_iters
    max_till_now = 0.0
    min_till_now = 1e6
    while step < num_iterations:
        model.train()
        running_loss = 0.0
        current_batch = 0
        if warmup_steps>0:
            for parameter in optimizer.param_groups:
                parameter['lr'] = learning_rate * min(
                    step/warmup_steps,1.0
                )
        for data in tqdm(train_loader, total=len(train_loader), leave=False):
            if config.data_set == "K-Mnist":
                images = data
            elif config.data_set == 'celeba':
                images = data['image'] 
                images = images.float() / 255.0
            else:
                images = data[0]
            images += 1.0
            t = torch.randint(1, N, (images.size(0),)).to(config.device)

            t_for_sde = t.view(-1, 1, 1, 1)

            sigma_t = sigmas[t].view(-1, 1, 1, 1)
            mu_t = mu[t].view(-1, 1, 1, 1)
            sigma_t_minus_1 = sigmas[t-1].view(-1, 1, 1, 1)
            mu_t_minus_1 = mu[t-1].view(-1, 1, 1, 1)

            images = images.to(config.device)
            x_now = images * torch.exp((mu_t - 0.5 * sigma_t ** 2) * config.delta * t_for_sde + torch.sqrt(
                config.delta * t_for_sde) * sigma_t * torch.randn_like(images).to(config.device))
            x_prev = torch.where(t_for_sde - 1 == 0,
                                 images,
                                 images * torch.exp((mu_t_minus_1 - 0.5 * sigma_t_minus_1 ** 2) * config.delta * (t_for_sde-1) + torch.sqrt(
                                     config.delta * (t_for_sde-1)) * sigma_t_minus_1 * torch.randn_like(images).to(config.device))
                                 )
            # Forward pass
            score = model(x_now, t)
            loss = torch.mean(
                new_loss(t+1, sigma_t, x_now, images, mu_t, score))
            loss /= grad_accum

            tqdm.write(f"Loss: {loss.item():.6f}")
            if local_rank == 0:
                if loss > max_till_now:
                    print("Loss is too high")
                    print(f"t: {t}")
                    config.summary_writer.add_scalar(
                        f"t/{ema_step}", t.min().item(), step)
                    max_till_now = loss
                if loss < min_till_now:
                    save_check_point(model.state_dict(), config.save_path,
                                     f"{config.save_path}/{config.run_name}_best.pth")
                    with open(f"{config.save_path}/bestmodellog.txt", "w") as f:
                        f.write(
                            f"Best model stored at {step} with loss {loss.item()}")
                    min_till_now = loss

            # Backward pass and optimize
            loss.backward()
            if (current_batch + 1) % grad_accum == 0 or current_batch == len(train_loader)-1:
                if grad_clip>=1.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                if config.multi_gpu_training:
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                running_loss += loss.item()

            print("batch:", current_batch)
            current_batch += 1
            step += config.num_gpus
        # Log loss to wandb
        # wandb.log({"loss": avg_loss})
        if local_rank == 0:
            avg_loss = running_loss / current_batch
            print(
                f" Epoch [{step + 1 }/{num_iterations}], Loss: {avg_loss:.4f}")
            # Log loss to wandb
            # wandb.log({"loss": avg_loss})
            config.summary_writer.add_scalar(f"Loss/train", avg_loss, step)

            if (step == config.num_gpus) or (step+ config.num_gpus) % 5000 == 0:
                check_point = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict()
                }
                save_check_point(
                    check_point, f"{config.save_path}/{step}", f"{config.save_path}/{step}/{config.run_name}_epoch={step}.pth")
                    
    # Save the trained model
    if local_rank == 0:
        check_point = {
            'step':step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        if not os.path.exists(config.save_path+'/'+str(step)):
            os.makedirs(config.save_path+'/'+str(step), exist_ok=True)
        save_check_point(check_point,config.save_path+'/'+str(step), f"{config.save_path}/{step}/{config.run_name}.pth")


