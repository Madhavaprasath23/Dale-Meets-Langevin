# from torch.utils.tensorboard import SummaryWriter
from configs.vp.cifar10_ddpmpp_deep_continuous import get_config
from configs.default_cifar10_configs import get_default_configs
from torchvision import transforms
from torch.optim import Adam
from torchvision import datasets
import ml_collections
import torch
from torch.utils.tensorboard import SummaryWriter
from models.ncsnpp import NCSNpp
#TODO Make Training Configs
def get_config_training():
    temp=get_default_configs()
    training_config = ml_collections.ConfigDict()

    training_config.config_module = get_config

    training_config.batch_size = temp.training.batch_size
    training_config.n_iters = temp.training.n_iters
    training_config.snapshot_freq = temp.training.snapshot_freq
    training_config.N = N = 1000
    training_config.sigma = sigmas =0.8* torch.ones(N)
    training_config.mu = 0.5 * sigmas ** 2
    training_config.save_path = "logs"
    training_config.transforms=transform = transforms.Compose([
                transforms.ToTensor(),
            ])
    training_config.model_class = NCSNpp
    training_config.train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform)
    training_config.multi_gpu_training=False
    training_config.batch_size = 128
    training_config.grad_batch_size = 128
    training_config.grad_accum = training_config.grad_batch_size//training_config.batch_size
    training_config.save_at_steps = 5000
    training_config.data_set = 'cifar10'
    training_config.run_name = "cifar10_runs"
    training_config.continue_model_path = None
    training_config.continue_model_step = 0
    training_config.summary_writer = None
    training_config.do_warmup_continue = False

    training_config.optim = ml_collections.ConfigDict()
    training_config.optim.opt = Adam
    training_config.optim.lr = temp.optim.lr
    training_config.optim.weight_decay = temp.optim.weight_decay
    training_config.optim.beta1 = temp.optim.beta1
    training_config.optim.eps = temp.optim.eps
    training_config.optim.warmup = temp.optim.warmup
    training_config.optim.grad_clip = temp.optim.grad_clip


    training_config.num_gpus=1
    training_config.delta = 1/N
    training_config.summary_log_dir='runs/cifar10'
    training_config.summary_writer = None

    return training_config  




def get_config_sampling(path):

    def update_configs(sampling_config):
        if sampling_config.sampler_mode == 'ums':
            if sampling_config.start_from_average:
                delta = 2e-4
                L = 4
                anneal_factor = 0.995
            else:
                delta = 0.00011421052631578947
                L = 6
                anneal_factor = 0.995
        elif sampling_config.sampler_mode == 'dls':
            delta = 0.00011421052631578947
            L = 6
            anneal_factor = 0.9995
        sampling_config.delta = delta
        sampling_config.L = L
        sampling_config.anneal_factor = anneal_factor
        return sampling_config
        

    sampling_config = ml_collections.ConfigDict()
    sampling_config.model_config = get_config
    sampling_config.model_class = NCSNpp
    sampling_config.path = path
    sampling_config.batch_size=batch_size = 32
    sampling_config.num_gpus=1
    sampling_config.data_set = 'cifar10'
    sampling_config.save_path = "logs"
    sampling_config.shape = (batch_size,3,32,32)
    sampling_config.num_samples = batch_size
    sampling_config.multi_gpu = False
    sampling_config.rank = 0
    sampling_config.show = True
    sampling_config.N = N = 1000
    sampling_config.sigma = sigmas = 0.8 * torch.ones(N)
    sampling_config.mu = 0.5 * sigmas ** 2
    sampling_config.constant = True
    sampling_config.do_tweedie = True
    sampling_config.sampler_mode = 'ums'
    sampling_config.mu_fit = [0.4075, 0.3785, 0.3693]
    sampling_config.sigma_fit =[0.8016, 0.8004, 0.8005]
    sampling_config.start_from_average = False

    update_configs(sampling_config)


    sampling_config.interactive = True
    sampling_config.show_progress = True
    sampling_config.show_intermediate = False
    sampling_config.show_text = False


    
    return sampling_config