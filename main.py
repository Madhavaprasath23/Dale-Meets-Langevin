import os
import warnings
os.environ['CUDA_VISIBLE_DEVICES']='1,2,3,4,5'

# Suppress TensorFlow C++ warnings and info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Suppress all Python warnings (e.g. TORCH_CUDA_ARCH_LIST UserWarning)
warnings.filterwarnings('ignore')

from utils import batch_save_tensors
from utils import class_average_image
from utils import *

import argparse
import os
import time

from gbm_configs.cond_ref_net.fmnist import get_config_training as get_config_training_cond_fmnist
from gbm_configs.cond_ref_net.mnist import get_config_training as get_config_training_cond_mnist
from gbm_configs.cond_ref_net.kmnist import get_config_training as get_config_training_cond_kmnist


from gbm_configs.NCSNDEEPER.fmnist import get_config_training as get_config_training_ncsndeeper_fmnist
from gbm_configs.NCSNDEEPER.mnist import get_config_training as get_config_training_ncsndeeper_mnist
from gbm_configs.NCSNDEEPER.kmnist import get_config_training as get_config_training_ncsndeeper_kmnist
from gbm_configs.NCSNV3.cifar10 import get_config_training as get_config_training_ncsnv3_cifar10

from gbm_configs.NCSNDEEPER.fmnist import update_configs as update_configs_fmnist_ncsn_deeper
from gbm_configs.NCSNDEEPER.mnist import update_configs as update_config_mnist_ncsn_deeper
from gbm_configs.NCSNDEEPER.kmnist import update_configs as update_config_kmnist_ncsn_deeper

from gbm_configs.cond_ref_net.fmnist import get_config_sampling as get_config_sampling_cond_fmnist
from gbm_configs.cond_ref_net.mnist import get_config_sampling as get_config_sampling_cond_mnist
from gbm_configs.cond_ref_net.kmnist import get_config_sampling as get_config_sampling_cond_kmnist
from gbm_configs.NCSNDEEPER.fmnist import get_config_sampling as get_config_sampling_ncsndeeper_fmnist
from gbm_configs.NCSNDEEPER.mnist import get_config_sampling as get_config_sampling_ncsndeeper_mnist
from gbm_configs.NCSNDEEPER.kmnist import get_config_sampling as get_config_sampling_ncsndeeper_kmnist
from gbm_configs.NCSNV3.cifar10 import get_config_sampling as get_config_sampling_ncsnv3_cifar10

from pathlib import Path
from mnist_exps.train_end_to_end import train_model
from mnist_exps.sampling import run_sampler
from mnist_exps.calculate_FID import calculate_fid_kid_tensor
from mnist_exps.nearest_neighbours import nearest_neighbours
from torch.distributed.launcher.api import LaunchConfig,elastic_launch

"""
adls: Annealed Dale Langevin Sampler
aums: Annealed Unconstrained Multiplicative Sampler
"""

def run_distributed(config,num_gpus,port,func):
    config_launch = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=num_gpus,
        rdzv_backend='c10d'
    )
    return elastic_launch(config_launch,func)(config)


def get_config(args):
    class_to_take = {
            'fmnist_cond_ref_net': get_config_training_cond_fmnist,
            'mnist_cond_ref_net': get_config_training_cond_mnist,
            'kmnist_cond_ref_net': get_config_training_cond_kmnist,
            'fmnist_ncsn_deeper': get_config_training_ncsndeeper_fmnist,
            'mnist_ncsn_deeper': get_config_training_ncsndeeper_mnist,
            'kmnist_ncsn_deeper': get_config_training_ncsndeeper_kmnist,
            'cifar10_ncsnv3': get_config_training_ncsnv3_cifar10}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = class_to_take.get(
                f'{args.dataset}_{args.model}',
                None)
    return config()

def get_sampler_config(args):
    class_to_take = {
            'fmnist_cond_ref_net': get_config_sampling_cond_fmnist,
            'mnist_cond_ref_net': get_config_sampling_cond_mnist,
            'kmnist_cond_ref_net': get_config_sampling_cond_kmnist,
            'fmnist_ncsn_deeper': get_config_sampling_ncsndeeper_fmnist,
            'mnist_ncsn_deeper': get_config_sampling_ncsndeeper_mnist,      
            'kmnist_ncsn_deeper': get_config_sampling_ncsndeeper_kmnist,
            'cifar10_ncsnv3': get_config_sampling_ncsnv3_cifar10}

    config = class_to_take.get(
                f'{args.dataset}_{args.model}',
                None)
    return config


def main():
    parser = argparse.ArgumentParser(description='Minimal GBM')

    # Core Arguments
    core_group = parser.add_argument_group('Core Configuration')
    core_group.add_argument('--mode', type=str, choices=['train', 'sample', 'calculate_fid_kid', 'calculate_fid', 'nearest_neighbors'], default='train', help='Execution mode: train, sample, evaluate FID/KID, or find nearest neighbors.')
    core_group.add_argument('--model', type=str, choices=['cond_ref_net', 'ncsn_deeper', 'ncsnv3'], default='cond_ref_net', help='Model architecture to use.')
    core_group.add_argument('--dataset', type=str, choices=['fmnist', 'mnist', 'kmnist', 'cifar10'], default='mnist', help='Dataset to train or sample from.')
    core_group.add_argument('--model_path', type=str, default=None, help='Path to an existing model checkpoint for sampling or resume.')

    # Sampling Arguments
    sample_group = parser.add_argument_group('Sampling Configuration')
    sample_group.add_argument('--sampler_mode', type=str, choices=['adls', 'aums'], default='adls', help='sampler mode to use (e.g., Dale-Langevin Sampler (adls) or Unconstrained Multiplicative Sampler (aums)).')
    sample_group.add_argument('--sampler_batch_size', type=int, default=2048, help='Batch size used during sampling.')
    sample_group.add_argument('--num_of_samples', type=int, default=50000, help='Total number of samples to generate.')
    sample_group.add_argument('--noise_start_point', type=str, choices=['class_averaged', 'noise'], default='noise', help='Initialization point for the diffusion sampling process.')
    sample_group.add_argument('--display_tweedie_images_intermediate', action='store_true', help='If set, visually display intermediate Tweedie generated images during sampling.')

    # Output/Saving Arguments
    io_group = parser.add_argument_group('I/O Configuration')
    io_group.add_argument('--save_path', type=str, default=None, help='Directory path to save generated images or outputs.')
    io_group.add_argument('--save_samples', action='store_true', help='If set, generated samples will be saved to disk.')
    io_group.add_argument('--save_intermediate_tensor', type=str, default='', help='Path to save the intermediate trajectory tensors.')

    # Distributed Training Arguments
    dist_group = parser.add_argument_group('Distributed Training Configuration')
    dist_group.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use. >1 uses PyTorch Distributed Data Parallel.')
    dist_group.add_argument('--port', type=int, default=29500, help='Master port for DDP distributed training.')

    args = parser.parse_args()

    if args.model == 'ncsnv3' and args.dataset != 'cifar10':
        raise ValueError("NCSNv3 can only be used with CIFAR10")
    if args.dataset == 'cifar10' and args.model != 'ncsnv3':
        raise ValueError("CIFAR10 can only be used with NCSNv3")

    config = get_config(args)
    if config is None:
        raise ValueError(f"Invalid dataset or model: {args.dataset} for {args.model}")
    if args.mode == 'train':
        if args.num_gpus > 1:
            config.num_gpus = args.num_gpus
            config.port = args.port
            config.multi_gpu_training = True
            run_distributed(config,args.num_gpus,args.port,train_model)
        else:
            config.multi_gpu_training = False
            train_model(config)


    elif args.mode == 'calculate_fid' or args.mode == 'sample':
        config = get_sampler_config(args)
        config = config(args.model_path)
        config.multi_gpu = args.num_gpus > 1
        config.sampler_mode = args.sampler_mode
        
        if args.model=='ncsn_deeper':
            if args.dataset=='mnist':
                update_config_mnist_ncsn_deeper(config)
            elif args.dataset=='kmnist':
                update_config_kmnist_ncsn_deeper(config)
            elif args.dataset=='fmnist':
                update_configs_fmnist_ncsn_deeper(config)
                

        config.batch_size= args.sampler_batch_size if args.num_of_samples >= args.sampler_batch_size else args.num_of_samples
        config.num_samples = args.num_of_samples
        config.save_path = args.save_path
        steps_to_sample = math.ceil(config.num_samples/(config.batch_size))
        config.show = False
        try:
            os.makedirs(config.save_path,exist_ok=True)
        except:
            config.save_path = f'logs/{args.dataset}_runs@{time.time()}_{args.model}/{config.num_samples}'
            os.makedirs(config.save_path)
        print("saving @",config.save_path)
        config.display_tweedie_images_intermediate = args.display_tweedie_images_intermediate
        config.start_from_average = False if args.noise_start_point == 'noise' else True
        if config.start_from_average and os.path.exists(f'logs/class_average/{config.data_set}'):
            os.makedirs(f'logs/class_average/{config.data_set}',exist_ok=True)
            class_averaged_images = class_average_image(config.data_set)
            batch_save_tensors(class_averaged_images,output_dir=f'logs/class_average/{config.data_set}')



        if args.num_gpus > 1:
            print(f"Running in distributed mode with {args.num_gpus} GPUs on port {args.port}")
            config.num_gpus = args.num_gpus
            config.port = args.port

            total_samples_gpu = run_distributed(config,args.num_gpus,args.port,run_sampler)

            
            #all gather the samples from different gpus and save
            print("saving")
            """total_samples = [torch.zeros_like(total_samples_gpu) for _ in range(config.num_gpus)]
            torch.distributed.all_gather(total_samples,total_samples_gpu)
            total_samples = torch.cat(total_samples,dim=0)"""
        else:
            total_samples = run_sampler(config)
        
        #total_samples = total_samples[:config.num_samples].detach().cpu()

        if args.mode == 'calculate_fid':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            fid,kid = calculate_fid_kid_tensor(
                config.data_set,
                total_samples,
                num_samples=args.num_of_samples,
                device=device
            )

            with open(os.path.join(config.save_path,'fid_kid.txt'),'w') as f:
                f.write(f'FID: {fid.item()}\n')
                f.write(f'KID: {kid.item()}\n')
        
        #if args.save_samples:
        #   batch_save_tensors(total_samples,config.save_path)

            
    elif args.mode == 'nearest_neighbors':
        #load the generated samples and calculate nearest neighbors from the training set
        config = get_sampler_config(args)
        config = config(args.model_path)
        config.start_from_average = False if args.noise_start_point == 'noise' else True
        config.update_configs(config)
        config.batch_size = 16
        config.num_samples = 16

        total_samples = run_sampler(config)
        if os.path.exists(config.save_path+'/nearest_neighbors') == False:
            os.makedirs(config.save_path+'/nearest_neighbors')
            os.makedirs(config.save_path+'/nearest_neighbors_images')
        
        batch_save_tensors(total_samples,config.save_path+'/nearest_neighbors_images',prefix='nearest_neighbors_sample_')
        


        for mode in ['inception','l2']:
            nearest_neighbours(
                path = config.save_path,
                data_set = config.data_set,
                num_samples = 10,
                device = device,
                mode = 'incept',
                pdf_name = f'Nearest_Neighbors_{config.sampler_mode}_{config.data_set}_{mode}.pdf',
            )
        os.rmdir(config.save_path+'/nearest_neighbors_images')


if __name__ == '__main__':
    main()
