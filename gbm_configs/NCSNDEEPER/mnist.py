from torchvision import transforms
from torch.optim import Adam
from torchvision import datasets
import ml_collections
import torch
from torch.utils.tensorboard import SummaryWriter
from models.ncsndeeper.ncsn import NCSNdeeper
from utils import *


def get_config_training():
    training_config = ml_collections.ConfigDict()
    training_config.transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    training_config.dataset=datasets.MNIST(
        root='./data',train=True,download=True,transform=training_config.transforms
    )
    training_config.multi_gpu_training=False
    training_config.batch_size = 128
    training_config.grad_batch_size = 128
    training_config.config_path = "yamal_configs/NCSNDEEPER/mnist.yml"
    training_config.config = get_config_path(training_config.config_path)
    training_config.model_class = NCSNdeeper
    training_config.grad_accum = training_config.grad_batch_size//training_config.batch_size
    training_config.save_at_steps = 5000
    training_config.data_set = 'mnist'
    training_config.optim = Adam
    training_config.run_name = "mnist_runs"
    training_config.continue_model_path = None
    training_config.continue_model_step = 0
    training_config.summary_writer = None
    training_config.do_warmup_continue = False
    training_config.num_gpus=1
    training_config.N = N = 1000
    training_config.delta = 1/N
    training_config.summary_log_dir='runs/mnist'
    training_config.summary_writer = SummaryWriter(log_dir=training_config.summary_log_dir)

    return training_config


def get_config_sampling(path):

    def update_configs(sampling_config):
        if sampling_config.sampler_mode == 'ums':
            delta = 0.000143844988828766
            L = 4
            anneal_factor = 0.995
        elif sampling_config.sampler_mode == 'dls':
            delta = 0.0006165454545454547
            L = 1
            anneal_factor = 0.995
        sampling_config.delta = delta
        sampling_config.L = L
        sampling_config.anneal_factor = anneal_factor

    sampling_config = ml_collections.ConfigDict()
    sampling_config.config_path = "yamal_configs/NCSNDEEPER/mnist.yml"
    sampling_config.config = get_config_path(sampling_config.config_path)
    sampling_config.model_class = NCSNdeeper
    sampling_config.path = path
    sampling_config.batch_size=batch_size = 32
    sampling_config.num_gpus=1
    sampling_config.data_set = 'mnist'
    sampling_config.save_path = "logs"
    sampling_config.shape = (batch_size,1,28,28)
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
    sampling_config.sampler_mode = 'ums'

    mu_fit,sigma_fit = 0.15651744604110718, 0.8352777361869812
    sampling_config.mu_fit = mu_fit
    sampling_config.sigma_fit = sigma_fit

    update_configs(sampling_config)
        
    sampling_config.interactive = True
    sampling_config.show_progress = True
    sampling_config.show_intermediate = False

    return sampling_config