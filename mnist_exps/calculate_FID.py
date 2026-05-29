
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmetrics.image.kid import KernelInceptionDistance
import torch
import numpy as np
from torchvision import datasets, transforms
from utils import *
from tqdm import tqdm
from torcheval.metrics import FrechetInceptionDistance
#from torchfidelity import calculate_metrics

class Calculate_metrics():
    def __init__(self,num_subsets=64, max_subset_size=64,device='cuda'):
        self.kid = KernelInceptionDistance(subsets=num_subsets,subset_size=max_subset_size).to(torch.device('cuda',0))
        self.fid = FrechetInceptionDistance().to(torch.device('cuda',0))

    def update_real_images(self,real):
        self.fid.update(real,is_real=True)

        real = real * 255.0
        real = real.to(dtype=torch.uint8,device = 'cuda')
        self.kid.update(real,real=True)
    def update_fake_images(self,fakes):
        self.fid.update(fakes,is_real=False)

        fakes = fakes * 255.0
        fakes = fakes.to(dtype=torch.uint8,device = 'cuda')
        self.kid.update(fakes,real=False)
    def calculate_metrics(self):
        return self.fid.compute(),self.kid.compute()

class FID_calc():
    def __init__(self,device='cuda'):
        self.fid = FrechetInceptionDistance(device  = device)
    

    def update_real_images(self,real):
        self.fid.update(real,is_real=True)
    def update_fake_images(self,fakes):
        self.fid.update(fakes,is_real=False)
    def calculate_metrics(self):
        return self.fid.compute()
    def reset(self):
        self.fid.reset()

def get_50k_images(path,start=0,end=50_000):
    images = []
    for i in range(start,end):
        image = Image.open(f'{path}/sample_{i}.png')
        images.append(np.array(image).astype(np.uint8))
    return images

def calculate_fid_kid(path,data_set='mnist',num_samples=50_000,device='cuda'):
    if data_set == 'mnist':
        test_dataset = datasets.MNIST(
            root='./data', train=False, download=True)
    elif data_set == 'celeb-a':
        test_dataset = CelebADataset(
            root='/home/nishanth/gbm2d/mnist_exps/data/img_align_celeba/celeba',
        )
    elif data_set == 'fashion-mnist':
        test_dataset = datasets.FashionMNIST(
            root='./data', train=False, download=True)
    elif data_set == 'cifar10':
        test_dataset = datasets.CIFAR10(
            root='./data', train=True, download=True,transform=transforms.ToTensor())
    elif data_set == 'k-mnist' or data_set=='kmnist':
        test_dataset = datasets.KMNIST(
            root='./data', train=False, download=True,transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=64, shuffle=False, num_workers=4)
        
    
    # Create a CalculateFID object
    
    fid_kid_calculator = Calculate_metrics(device=device)
    # Iterate through the DataLoader and update the FID and KID scores
    num_iterations = num_samples // 64
    print(len(test_loader.dataset),num_iterations)

    start=0
    end = 64
    for iterations in tqdm(range(num_iterations)):
        for real_images,_ in test_loader:

            real_images = real_images
            real_images_kid = real_images* 255.0
            real_images_kid = real_images.to(dtype=torch.uint8,device = device) 
            # Update the FID and KID scores with real images
            if data_set!='cifar10':
                real_images = real_images.repeat(1,3,1,1)
            fid_kid_calculator.update_real_images(real_images)
            
            
            fake_images  = [torch.tensor(i) for i in get_50k_images(path=path,start=start,end=start + real_images.shape[0])]
            start += real_images.shape[0]
            fake_images = torch.stack(fake_images).unsqueeze(1)
            fake_images = fake_images.to(dtype=torch.uint8,device=device)
            fake_images = fake_images.repeat(1,3,1,1)
            fid_kid_calculator.update_fake_images(fake_images)
    print("Calculating FID and KID scores...")
    print("FID and KID",fid_kid_calculator.calculate_metrics())



def calculate_fid_kid_tensor(fake_images_tensor,data_set='mnist',device='cuda',batch_size = 64,num_samples=5_000):
    if data_set == 'mnist':
        test_dataset = datasets.MNIST(
            root='./data', train=False, download=True,transform=transforms.ToTensor())
    elif data_set == 'fashion-mnist':
        test_dataset = datasets.FashionMNIST(
            root='./data', train=False, download=True,transform=transforms.ToTensor())
    elif data_set == 'cifar10':
        test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True,transform=transforms.ToTensor())
    elif data_set == 'k-mnist' or 'kmnist':
        test_dataset = datasets.KMNIST(
            root='./data', train=False, download=True,transform=transforms.ToTensor())
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    

    fid_calculator = Calculate_metrics(device=device)
    num_iterations = num_samples // len(test_loader.dataset)
    
    start = 0
    end = batch_size
    if num_iterations>=1:
        for iterations in tqdm(range(num_iterations)):
            for idx, real_images in enumerate(test_loader):
                if data_set != 'k-mnist':
                    real_images = real_images[0]
                real_images = real_images.to(device)
                # Update the FID and KID scores with real images
                if data_set!='cifar10':
                    real_images = real_images.repeat(1,3,1,1)
                fid_calculator.update_real_images(real_images)
                
                fake_images = fake_images_tensor[start:end].to(device)
                if data_set!='cifar10':
                    fake_images = fake_images.repeat(1,3,1,1)
                fid_calculator.update_fake_images(fake_images)
                start += batch_size
                end += batch_size
    else:
        for real_images in test_loader:
            if data_set !='k-mnist':
                real_images = real_images[0]
            real_images = real_images
            real_images = real_images.to(device)
            # Update the FID and KID scores with real images
            if data_set!='cifar10':
                real_images = real_images.repeat(1,3,1,1)
            fid_calculator.update_real_images(real_images)
            
            fake_images = fake_images_tensor[start:end].to(device)
            if data_set !='cifar10':
                fake_images = fake_images.repeat(1,3,1,1)
            fid_calculator.update_fake_images(fake_images)
            start += batch_size
            end += batch_size
            if start>=num_samples:
                break
    return fid_calculator.calculate_metrics()

