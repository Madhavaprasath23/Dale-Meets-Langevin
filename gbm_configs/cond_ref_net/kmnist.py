from utils import K_Mnist
from torchvision import transforms
from torch.optim import Adam
from torchvision import datasets
import ml_collections
import torch
from torch.utils.tensorboard import SummaryWriter
from cond_refinenet_dilated import ConditionalResidualBlock
from utils import *

def get_config_training():
    training_config = ml_collecions.ConfigDict()
    training_config.transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    training_config.transforms=K_Mnist(path='mnist_exps/data/K-mnist/kmnist-train-imgs.npz',transform=training_config.transforms)
    training_config.multi_gpu_training-False
    training_config.n_iters = 100000
    training_config.snapshot_freq = 5000
    training_config.N = N = 1000
    training_config.sigma = sigmas =0.8* torch.ones(N)
    training_config.mu = 0.5 * sigmas ** 2
    training_config.batch_size = 128
    training_config.save_path = 'logs'

    
    training_config.grad_batch_size = 128
    training_config.model_class = ConditionalResidualBlock
    training_config.config = get_config_path(training_config.config_path)
    training_config.grad_accum = training_config.grad_batch_size//training_config.batch_size
    training_config.save_at_steps = 5000
    training_config.data_set = 'k-mnist'
    training_config.optim = Adam
    training_config.run_name = "k-mnist_runs"
    training_config.continue_model_path = None
    training_config.continue_model_step = 0
    training_config.summary_writer = None
    training_config.do_warmup_continue = False
    training_config.num_gpus=1
    training_config.delta = 1/N
    training_config.summary_log_dir='runs/k-mnist'
    training_config.summary_writer = SummaryWriter(log_dir=training_config.summary_log_dir)

    return training_config

def get_config_sampling(path,device):

    sampling_config = ml_collections.ConfigDict()
    sampling_config.config_path = "yamal_configs/cond_ref_net/anneal_mnist.yml"
    sampling_config.model_class = ConditionalResidualBlock
    sampling_config.config = get_config_path(sampling_config.config_path)
    sampling_config.path = path
    sampling_config.batch_size=batch_size = 32
    sampling_config.num_gpus=1
    sampling_config.device = device
    sampling_config.data_set = 'k-mnist'
    sampling_config.save_path = "logs"
    sampling_config.shape = (batch_size,3,32,32)
    sampling_config.delta = 2e-4
    sampling_config.num_samples = batch_size
    sampling_config.multi_gpu = False
    sampling_config.rank = 0
    sampling_config.show = True
    sampling_config.N = N = 1000
    sampling_config.sigma = sigmas = 0.8 * torch.ones(N)
    sampling_config.mu = 0.5 * sigmas ** 2
    sampling_config.L = 4
    sampling_config.constant = True
    sampling_config.do_tweedie = False

    mu_fit,sigma_fit = 0.2140171080827713, 0.8290426135063171
    sampling_config.mu = mu_fit
    sampling_config.sigma = sigma_fit
    sampling_config.interactive = True
    sampling_config.show_progress = True
    sampling_config.show_intermediate = False

    return sampling_config