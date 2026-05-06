# from torch.utils.tensorboard import SummaryWriter
from configs.ve.celeba_ncsnpp import get_config
from torchvision import transforms
from torch.optim import Adam
from torchvision import datasets
import ml_collections
import torch
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset,load_from_disk
#TODO Make Training Configs

def get_CelebaHQ(path):
    ds = load_from_disk(path).with_format("torch")
    return ds

def get_config_training():
    temp=get_config()
    training_config = ml_collections.ConfigDict()

    training_config.config_module = get_config

    training_config.batch_size = temp.training.batch_size
    training_config.n_iters = temp.n_iters
    training_config.snapshot_freq = temp.snapshot_freq
    training_config.N = N = 1000
    training_config.sigma = sigmas =0.8* torch.ones(N)
    training_config.mu = 0.5 * sigmas ** 2
    training_config.save_path = "logs"
    training_config.transforms=transform = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
            ])
    
    """Get the data or either load them while training"""
    try:
        training_config.train_dataset = get_CelebaHQ("data/celebaHQdata/train")
    except:
        ds = load_dataset("eurecom-ds/celeba-hq-256")
        ds.save_to_disk("data/celebaHQ")
        training_config.train_dataset = get_CelebaHQ("data/celebaHQdata/train")
    
    training_config.multi_gpu_training=False
    training_config.batch_size = 64
    training_config.grad_batch_size = 128
    training_config.grad_accum = training_config.grad_batch_size//training_config.batch_size
    training_config.save_at_steps = 5000
    training_config.data_set = 'celeba'
    training_config.optim = Adam
    training_config.run_name = "celeba_runs"
    training_config.continue_model_path = None
    training_config.continue_model_step = 0
    training_config.summary_writer = None
    training_config.do_warmup_continue = False
    training_config.num_gpus=1
    training_config.delta = 1/N
    training_config.summary_log_dir='runs/celeba_runs'
    training_config.summary_writer = SummaryWriter(log_dir=training_config.summary_log_dir)
    return training_config


def get_config_sampling(path,device):

    sampling_config = ml_collections.ConfigDict()
    sampling_config.model_config = get_config
    sampling_config.path = path
    sampling_config.batch_size=batch_size = 32
    sampling_config.num_gpus=1
    sampling_config.device = device
    sampling_config.data_set = 'celeba'
    sampling_config.save_path = "logs"
    sampling_config.shape = (batch_size,3,64,64)
    sampling_config.delta = 2e-4
    sampling_config.num_samples = batch_size
    sampling_config.multi_gpu = False
    sampling_config.rank = 0
    sampling_config.show = True
    sampling_config.N = N = 1000
    sampling_config.sigma = sigmas = 0.8 * torch.ones(N)
    sampling_config.mu = 0.5 * sigmas ** 2
    sampling_config.L = 4

    return sampling_config