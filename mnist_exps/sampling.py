
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image
import math
import torch
from utils import *
from pathlib import Path

from gbm_configs.NCSNV3.cifar10 import get_config_sampling as get_config_sampling_ncsnv3_cifar10

"""
adls: Dale Langevin Sampler
aums: Unconstrained Multiplicative Sampler
"""

def convert_ddp_to_normal(state_dict):
    parameters = {}
    for name, param in state_dict.items():
        if name.startswith('module.'):
            name = name[7:]
        parameters[name] = param
    return parameters


def load_class_avg_tensors(folder, to_device: str = 'cpu'):
    """Read saved class-average PNGs from `folder` and return a list of torch.FloatTensors

    Each tensor is shaped (C, H, W), dtype=torch.float32 and normalized to [0,1].
    Files are read in lexicographic order.
    """
    pngs = sorted([p for p in folder.iterdir() if p.suffix.lower() in ('.png',) and p.name != 'grid.png'])
    if not pngs:
        raise FileNotFoundError(f"No PNG files found in {folder}")
    tensors = []
    for p in pngs:
        with Image.open(p) as im:
            
            if im.size[0] == 28 and im.size[1] == 28:
                im = im.convert('L')
            else:
                im = im.convert('RGB')
            arr = np.array(im, dtype=np.float32) / 255.0
            if arr.ndim == 2:  # grayscale
                t = torch.from_numpy(arr).unsqueeze(0).contiguous()  # Shape: (1, H, W)
            else:  # RGB
                t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            # HWC -> CHW
            tensors.append(t)
    device = torch.device(to_device)
    tensors = [t.to(device) for t in tensors]
    return tensors


def make_lognormal_noise(mean, std_dev, nosiy_image, device='cuda'):
    """
    Generates lognormal noise based on the provided mean and standard deviation.

    Args:
        mean (tensor): Mean of the lognormal distribution.
        std_dev (tensor): Standard deviation of the lognormal distribution.
        images (tensor): Input images to which noise will be added.
        device (str): Device to perform the computation on.

    Returns:
        tensor: Lognormal noise tensor.
    """
    # Generate lognormal noise
    noise = nosiy_image
    log_normal_noise = torch.cat([torch.exp(mean[0] + std_dev[0] * noise[:, 0, ...].unsqueeze(1)),
                                  torch.exp(mean[1] + std_dev[1] *
                                            noise[:, 1, ...].unsqueeze(1)),
                                  torch.exp(mean[2] + std_dev[2] * noise[:, 2, ...].unsqueeze(1))], dim=1)

    print(
        f"Lognormal Noise: {log_normal_noise.min()}, {log_normal_noise.max()}")
    return log_normal_noise


def sample_one_step_no_hacks(test_image, model, device, save_path, mu, sigmas, N, delta,mu_fit,sigma_fit, batch_size=64, data_set='mnist', show=True, L=1,constant=False,current_step=0,sampler_mode='adls',do_tweedie=False,config=None,show_text=False):
    """
    Generate samples from a trained score-based model using one-step 
    iterative refinement without additional heuristics.

    The function starts from pure noise (log-normal for non-CIFAR datasets, 
    channel-wise log-normal noise for CIFAR-10) and iteratively updates 
    samples using the learned score function over `N` steps. Intermediate 
    results and final generated images can be visualized and saved.

    Args:
        test_image (torch.Tensor):
            Reference tensor with the same shape as the target dataset 
            (used to determine shape for initial noise).
        model (nn.Module):
            Trained score-based generative model.
        device (str or torch.device):
            Device to run sampling on.
        save_path (str):
            Directory path where generated images and logs are saved.
        mu (torch.Tensor):
            Drift schedule used for sampling (1D tensor of length `N`).
        sigmas (torch.Tensor):
            Noise schedule used for sampling (1D tensor of length `N`).
        N (int):
            Total number of sampling steps.
        delta (float):
            Step size for the update rule (SDE discretization).
        batch_size (int, optional, default=64):
            Batch size used during sampling.
        data_set (str, optional, default='mnist'):
            Name of the dataset. Determines the type of initial noise
            (`'cifar10'` uses channel-wise log-normal noise).
        show (bool, optional, default=True):
            Whether to visualize intermediate steps with `show_grid`.

    Returns:
        torch.Tensor:
            Final generated samples as a tensor in range [0, 1].

    Notes:
        - For non-CIFAR datasets, initial noise is sampled from a log-normal
          distribution with fitted parameters.
        - For CIFAR-10, per-channel log-normal noise is used.
        - If NaNs appear during sampling, an error message is written to
          `error.txt` in the save path.
        - Intermediate and final samples are saved using `show_grid`.
    """
    config.rank = dist.get_rank() if config.multi_gpu else 0
    if show_text:
        config.show_progress = False
    
    print(f"Starting sampling with sampler_mode: {config.sampler_mode}, delta {delta}, L {L}, dotweedie: {do_tweedie} class_average start: {config.start_from_average}")
    with torch.no_grad():
        if data_set != 'cifar10':
            if config.start_from_average:
                x_now = load_class_avg_tensors(Path(f'logs/class_average_images/{data_set.lower()}'), to_device=device)
                x_now = torch.stack(x_now)
                num_classes = x_now.shape[0]
                x_now = x_now.unsqueeze(0).repeat(batch_size//num_classes, 1, 1,1, 1)
                x_now = x_now.view(-1,1,28,28)
                if x_now.shape[0] < batch_size:
                    remaining = batch_size - x_now.shape[0]
                    x_now = torch.cat((x_now,x_now[:remaining]),dim=0)
                    print(f"Repeating {remaining} samples to match batch size")
                
                x_now = x_now + 1.0
                t = torch.full((x_now.shape[0],), 999,
                                    dtype=torch.long).to(device)
                
                x_now = forward_sde_gbm_fast(mu=torch.tensor(mu_fit).to(device),sigma=torch.tensor(sigma_fit).to(device),x_now=x_now,t=torch.tensor([999]).to(device),N=N,device=device)
                
                x_now = x_now.to(device)

            else:
                mu_fit, sigma_fit = mu_fit,sigma_fit
                x_now = torch.exp(mu_fit + sigma_fit *
                                torch.randn_like(test_image).to(device)).float()

                if show:
                    show_grid(x_now.detach().cpu(), title='Pure Noise', step=0,
                            dataset=data_set, save_path=save_path, gen=True)

        else:
            if config.start_from_average:
                x_now = torch.randn_like(test_image).to(device)
                mean_list = mu_fit
                std_list = sigma_fit
                x_now = load_class_avg_tensors(Path(f'logs/class_average_images/{data_set.lower()}'), to_device=device)
                x_now = torch.stack(x_now)
                num_classes = x_now.shape[0]
                x_now = x_now.unsqueeze(0).repeat(batch_size//num_classes, 1, 1,1, 1)
                x_now = x_now.view(-1,3,32,32)
                if x_now.shape[0] < batch_size:
                    remaining = batch_size - x_now.shape[0]
                    print(f"Repeating {remaining} samples to match batch size")
                    x_now = torch.cat((x_now,x_now[:remaining]),dim=0)
                
                x_now = x_now + 1.0
                x_now = forward_sde_gbm_channelwise(mu=torch.tensor(mean_list),sigma=torch.tensor(std_list),x_now=x_now,t=torch.tensor([999]),N=N,device=device)
                x_now = x_now.to(device)
            else:
                x_now = torch.randn_like(test_image).to(device)
                mean_list = mu_fit
                std_list = sigma_fit
                x_now = make_lognormal_noise(mean = mean_list,std_dev=std_list,nosiy_image=x_now,device=device)

        sigma_t = sigmas[999]
        mu_t = 0.5 * (sigma_t ** 2)
        if show :
            temp = torch.clamp(x_now.clone(), 1.0, 2.0) - \
                            1.0  # work on this
            show_grid(temp.detach().cpu(), save_path=save_path,
                    title="Original Images", step=0, dataset=data_set, gen=True)
        if show:
            temp = torch.clamp(x_now.clone(), 1.0, 2.0) - \
                            1.0  # work on this
            show_grid(temp.detach().cpu(), save_path=save_path,
                    title="Noisy Images", step=0, dataset=data_set, gen=True)
        anneal = 1.0
        for t in tqdm(range(N),total=N, desc='Sampling') if config.show_progress else range(N):
            delta, L,factor = get_delta_factor_per_step(config, N-t-1)
            x_now = x_now.to(device, dtype=torch.float32)
            current_t = torch.full((batch_size,), N-t-1,
                                   dtype=torch.long).to(device, dtype=torch.long)
            score = model(x_now.float(), current_t)
            if (t+1) % 100 == 0 or (900 <= t and (t+1) % 10 == 0) or ((N-t-1) <= 9):
                if show_text:
                    print(
                        f"Score: {score.min()}, {score.max()}, {torch.norm(score)}")
                if show:
                    show_grid(score.detach().cpu(), save_path=save_path,
                              title="Score", step=N - (t+1), dataset=data_set, gen=True)

            sigma_t = sigmas[N-t-1]
            mu_t = mu[N-t-1]
            if sampler_mode == 'adls':
                x_now,score = adls_sampler(x_now,current_t,delta,sigma_t,mu_t,anneal,model,device,L,N,t,show_text=show_text)
            else:
                x_now,score = aums_sampler(x_now,current_t,delta,sigma_t,mu_t,anneal,model,device,L,N,t,show_text=show_text)

            anneal = anneal * factor
            if torch.sum(torch.isnan(x_now)):
                with open(save_path+'/error.txt', 'w') as f:
                    f.write(
                        f"Error at step {N - (t+1)} with delta {delta}")
            if show:
                if (t+1) % 100 == 0 or (900 <= t and (t+1) % 10 == 0) or ((N-t-1) <= 9):
                
                    temp = x_now.clone()
                    print(f"Before clamping: Maximum:{temp.max()} Minimum:{temp.min()}")
                    temp = torch.clamp(temp,1.0,2.0) - 1.0
                    print(f"After clamping : Maximum:{temp.max()}Minimum:{temp.min()}")
                    show_grid(temp.detach().cpu(), save_path=save_path,
                              title=f"Generated Images", step=N-(t+1), dataset=data_set, gen=True)
            if show_text:
                print(
                    f"Step {N - (t+1)}: {x_now.min()}, {x_now.max()}, {delta}")
        if show:
            temp = x_now.clone()
            print(f"Before clamping: Maximum:{temp.max()} Minimum:{temp.min()}")
            temp = torch.clamp(temp,1.0,2.0) - 1.0
            print(f"After clamping : Maximum:{temp.max()}Minimum:{temp.min()}")
            show_grid(temp.detach().cpu(), save_path=save_path,
                      title=f"Generated Images", step=N-(t+1), dataset=data_set, gen=True)
        if do_tweedie:
            for step in tqdm(range(config.tweedie_steps), desc='Tweedie') if config.show_progress else range(25):
                current_t = torch.full((batch_size,), 0,
                                    dtype=torch.long).to(device, dtype=torch.long)
                score = model(x_now.float(),current_t)
                if config.sampler_mode == 'adls':
                    x_now = adls_denoiser(x_now, score, delta, sigma_t, mu_t)
                else:
                    x_now = aums_denoiser(x_now, score, delta, sigma_t, mu_t)
                
                temp = x_now.clone().cpu()
                if config.display_tweedie_images_intermediate:
                    temp= torch.clamp(x_now,1.0,2.0) - 1.0
                    if os.path.exists(save_path+'/tweedie') == False:
                        os.makedirs(save_path+'/tweedie')
                    
                    save_image(
                        temp.detach().cpu(), f"{save_path}/tweedie/tweedie_{step}.png", normalize=True
                    )
                    batch_save_tensors(temp.detach().cpu(), save_path=save_path+'/tweedie', prefix=f'tweedie_{step}_')
                if show:
                    temp = torch.clamp(x_now.clone(), 1.0, 2.0) - \
                            1.0
                    show_grid(temp.detach().cpu(), save_path=save_path,
                        title=f"Tweedie", step=step, dataset=data_set, gen=True)
                    
                    temp = torch.clamp(score.clone(),1.0,2.0) - 1.0
                    show_grid(
                        temp.detach().cpu(), save_path=save_path,
                        title=f"Tweedie_score", step=step, dataset=data_set, gen=True
                    )
        temp = torch.clamp(x_now.clone(), 1.0, 2.0) - \
                1.0
        return temp


def get_delta_factor_per_step(config,current_step):
    if config.constant:
        return config.delta, config.L, config.anneal_factor
    else:
        if current_step > 100:
            delta = 0.001
            factor = 1.0
            L = 1
        elif current_step > 50:
            delta = 0.0005
            factor = 0.9995
            L=2
        elif current_step >25:
            delta = 0.0006
            factor = 0.9993
            L=3
        else:
            delta = 0.0005
            factor = 0.999
            L=3
        return delta, L,factor


def optimal_sampler(config):
    """
    Generate samples from a trained NCSN++ model using an optimal
    score-based sampling procedure.

    This function:
        - Loads a pretrained NCSN++ model from checkpoint.
        - Sets up fixed noise and drift schedules.
        - Iteratively generates samples in batches using 
          `sample_one_step_no_hacks`.
        - Optionally visualizes intermediate outputs with `show_grid`.
        - Concatenates and returns all generated samples.

    Args:
        config: Configuration object with the following expected fields:
            - save_path (str): Directory to save generated outputs.
            - model_config (callable): Function returning model config.
            - path (str): Path to the trained model checkpoint.
            - rank (int): Local rank / GPU index.
            - num_gpus (int): Number of GPUs available.
            - device (str or torch.device): Target device.
            - shape (tuple): Shape of the input tensor for noise initialization.
            - show (bool): Whether to visualize intermediate samples.
            - data_set (str): Dataset name (used for visualization).
            - num_samples (int): Total number of samples to generate.
            - batch_size (int): Batch size for generation.
            - delta (float): SDE step size for the sampler.

    Returns:
        torch.Tensor:
            Generated samples concatenated across batches.

    Notes:
        - Model weights are automatically converted from DDP format if needed.
        - If `num_gpus > 1`, the model is wrapped with `torch.nn.DataParallel`.
        - A fixed noise schedule (`sigmas`, `mu`) with `N=1000` steps is used.
        - Intermediate and final samples are visualized with `show_grid` if `show=True`.
    """
    if config.multi_gpu:
        dist.init_process_group(backend='nccl')

    config.rank = dist.get_rank() if config.multi_gpu else 0
    config.device = torch.device(f'cuda:{config.rank}' if torch.cuda.is_available() else 'cpu')


    if config.save_path:
        if config.show:
            os.makedirs(config.save_path+'/gen', exist_ok=True)
        else:
            os.makedirs(config.save_path, exist_ok=True)

    if config.data_set.lower() not in ['fmnist', 'fashion-mnist', 'mnist', 'K-mnist','kmnist']:
        model = config.model_class(config=config.model_config()).to(config.rank)
    else:
        model_config = config.config
        model_config.device = config.device
        model = config.model_class(config=model_config)
    
    sd = torch.load(config.path,map_location=config.device)
    try:
        sd_n = sd['model_state_dict']
    except:
        sd_n = sd
    sd_w = convert_ddp_to_normal(sd_n)
    try:
        model.load_state_dict(sd_w)
    except:
        model.load_state_dict(sd_w)
    model = model.to(config.device)

    print(f"Model loaded on device {config.device},{config.num_gpus>1} GPUs available, multi_gpu: {config.multi_gpu}")
    if config.num_gpus > 1:
        model = DDP(model,device_ids=[config.rank],output_device=config.rank) if config.multi_gpu else model.to(config.device)
    model.eval()
    print("completed model setup")
    N = 1000

    sigmas = 0.8 * torch.ones(N).to(config.device)
    mu = 0.5 * sigmas ** 2
    total_samples = []
    steps_to_sample = math.ceil(config.num_samples/(config.batch_size))
    delta = config.delta
    L = config.L
    constant = config.constant
    show = config.show
    sampler_mode = config.sampler_mode

    save_path = config.save_path
    batch_size = config.batch_size
    do_tweedie = config.do_tweedie
    
    _,c,h,w = config.shape
    config.shape = (batch_size,c,h,w)
    test_image = torch.randn(config.shape).to(config.device)


    for cur_step in range(steps_to_sample):
        temp = sample_one_step_no_hacks(test_image=test_image, model=model, device=config.device, mu=mu, sigmas=sigmas,data_set=config.data_set,
                                        N=N, delta=delta, save_path=save_path, batch_size=batch_size, show=show, L=L,constant=constant,current_step=cur_step,do_tweedie=do_tweedie,sampler_mode=sampler_mode,
                                       mu_fit=config.mu_fit,sigma_fit=config.sigma_fit,config=config,show_text=config.show_intermediate)
        config.rank = dist.get_rank() if config.multi_gpu else 0

        
        if temp is None:
            return None
        if total_samples == []:
            total_samples = temp
        else:
            total_samples = torch.cat((total_samples, temp), dim=0)
    return total_samples.detach().cpu()

def adls_sampler(x_now,current_t,delta,sigma_t,mu_t,anneal,model,device,L,N,t,show_text):
    """Mode 1 - GBM iterative score-based sampling.

    Wraps `optimal_sampler`. All parameters are read from `config`.

    Args:
        config: Sampling configuration object (see `optimal_sampler` for fields).
        constant (bool): If True, use a constant delta/L schedule.

    Returns:
        torch.Tensor: Generated samples.
    """
    for _ in range(L):
        score = model(x_now, current_t)
        x_now = x_now * torch.exp(delta * (sigma_t ** 2) * (x_now * score) - delta * (mu_t - 1.5 * (sigma_t**2)) * torch.ones_like(x_now) +
                                  anneal * math.sqrt(delta) * sigma_t * torch.randn_like(x_now).to(device))
        if show_text:
            print(f"t: {N - (t+1)}, delta: {delta}, anneal: {anneal}")
    return x_now,score



def adls_denoiser(x_now,score,delta,sigma_t,mu_t):
    return x_now * torch.exp(delta * (sigma_t ** 2) * (x_now * score) - delta * (mu_t - 1.5 * (sigma_t**2)) * torch.ones_like(x_now))
   

def aums_denoiser(x_now,score,delta,sigma_t,mu_t):
    return x_now * ((1+2*delta*sigma_t**2) * torch.ones_like(x_now) #(1 + 2δσ2)1 

    - (delta*mu_t * torch.ones_like(x_now)) + #δμ
        
        (delta*sigma_t**2 * (x_now * score)))

def aums_sampler(x_now,current_t,delta,sigma_t,mu_t,anneal,model,device,L,N,t,show_text):
    for _ in range(L):
        score = model(x_now, current_t)
        
        x_now = x_now * ((1+2*delta*sigma_t**2) * torch.ones_like(x_now) #(1 + 2δσ2)1 

        - (delta*mu_t * torch.ones_like(x_now)) + #δμ
        
        (delta*sigma_t**2 * (x_now * score)) + #δσ2Xk ◦ ∇ log pXt (Xk, k) 
        (anneal * math.sqrt(delta) * sigma_t * torch.randn_like(x_now).to(device)) # √δσZk

        )
        """if (N-t-1)<=break_step:
            maximum = x_now.max()
            x_now = torch.clamp(x_now,1.0,maximum)"""

        if show_text:
            print(f"t: {N - (t+1)}, delta: {delta}, anneal: {anneal}")
    return x_now,score
    




def run_sampler(config):
    """Run the sampler for the requested mode.

    Args:
        config: Sampling configuration object.
        sampler_mode (str): 'dls' or 'aums'.
        constant (bool): Passed to mode 1 only.

    Returns:
        torch.Tensor: Generated samples.
    """
    if config.sampler_mode in ['adls','aums']:
        return optimal_sampler(config)
    else:
        raise ValueError(f"Unknown sampling mode: {config.sampler_mode}. Choose 'adls' or 'aums'.")
