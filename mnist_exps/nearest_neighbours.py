import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.models import Inception3,Inception_V3_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import sys
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

class NearestNeighbours(nn.Module):
    def __init__(self,device='cuda'):
        super(NearestNeighbours, self).__init__()
        self.inception = Inception3(aux_logits=False,init_weights=Inception_V3_Weights)
        self.inception.aux_logits = False
        self.inception.to(device)
        self.inception.fc = nn.Identity()  

    def forward(self, x):
        features = self.inception(x)
        return features

def get_nearest_neighbours(features, generated_features, num_samples = 10,mode='inception'):
    # Compute pairwise distances
    if mode == 'inception':
        dists = torch.cdist(generated_features.view(-1,2048),features.view(-1,2048), p=2)
    else:
        dists = torch.cdist(generated_features.view(-1,28*28),features.view(-1,28*28), p=2)
    
    # Get the indices of the nearest neighbours
    _, indices = torch.topk(dists, num_samples, largest=False)
    
    
    return indices

def take_first_n_images(path, n):
    from PIL import Image
    import os

    images = []
    list_random = np.random.randint(20_000, size=(10))
    def get_images(random_index):
        return f'sample_{random_index}.png'
    image_path=[get_images(i) for i in list_random.tolist()]
    for filename in image_path:
        if filename.endswith('.png'):
            img_path = os.path.join(path, filename)
            img = Image.open(img_path)
            images.append(img)
            if len(images) == n:
                break
    return images
def nearest_neighbours(path,save_path,data_set = 'mnist',num_samples = 10,device='cuda',mode='inception'):

    # Load the dataset
    if data_set == 'mnist':
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor
        dataset = MNIST(root='./data', train=True, download=True, transform=transforms.Compose([ToTensor()]))
    elif data_set == 'fmnist' or data_set=='fashion-mnist':
        from torchvision.datasets import FashionMNIST
        from torchvision.transforms import ToTensor
        dataset = FashionMNIST(root='./data', train=True, download=True, transform=transforms.Compose([ToTensor()]))
    elif data_set == 'cifar10':
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import ToTensor
        dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([ToTensor()]))
    elif data_set == 'k-mnist' or data_set=='kmnist':
        from torchvision.transforms import ToTensor
        dataset = K_Mnist(path="mnist_exps/data/K-mnist/kmnist-train-imgs.npz")
    else:
        raise ValueError("Unsupported dataset")

    # Create a DataLoader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    # Initialize the model
    model = NearestNeighbours(device = device)
    model.eval()

    all_features = []
    if mode == 'inception':
        with torch.no_grad():
            for data in tqdm(data_loader,total=len(data_loader)):
                if data_set!= 'k-mnist':
                    images, _ = data
                else:
                    images = data
                images = transforms.Compose([
                    transforms.Resize((299, 299))
                ])(images)
                images = images.to(device)
                if data_set!='cifar10':
                    images = images.repeat(1, 3, 1, 1)  # Repeat the image tensor to match Inception's input shape
                print(images.shape)
                features = model(images)
                all_features.append(features)
    else:
        with torch.no_grad():
            for data in tqdm(data_loader,total=len(data_loader)):
                if data_set != 'k-mnist':
                    images, _ = data
                else:
                    images = data
                print(images.shape)
                images = transforms.Compose([
                    transforms.Resize((28,28))
                ])(images)
                images = images.to(device)
                all_features.append(images)

    all_features = torch.cat(all_features, dim=0)
    all_features = all_features.to(device)

    generated_images = take_first_n_images(path, num_samples)
    
    if mode == 'inception':
        generated_samples = torch.stack([transforms.Compose([
            ToTensor(), transforms.Resize((299, 299))
        ])(img) for img in generated_images])
        if data_set!='cifar10':
            generated_samples = generated_samples.repeat(1, 3, 1, 1)  
        generated_samples = generated_samples.to(device)
        generated_samples = model(generated_samples)
    else:
        generated_samples = torch.stack([transforms.Compose([
            ToTensor(), transforms.Resize((28,28))
        ])(img) for img in generated_images])
        generated_samples = generated_samples.repeat(1, 1, 1, 1)
        generated_samples = generated_samples.to(device)
    indices = get_nearest_neighbours(all_features, generated_samples, num_samples,mode=mode)
    im=[]
    for i in range(len(indices)):
        mat=[]
        for j in range(len(indices[i])):
            if data_set != 'k-mnist' or data_set!='kmnist':
                image = data_loader.dataset[indices[i][j].item()]
            else:
                image = data_loader.dataset[indices[i][j].item()]
            image = transforms.Compose([
                transforms.Resize((28,28))])(image)
            mat.append(image.permute(1,2,0))
        im.append(mat)
    

    # Save the images with generated samples


    index =0 
    fig,ax = plt.subplots(10,11,figsize=(num_samples,10))
    for i in range(len(im)):
        img = im[i]
        ax[i,0].imshow(np.array(generated_images[index]),cmap='gray')
        ax[i,0].axis('off')
        for j in range(1, 11):
            ax[i,j].imshow(img[j-1], cmap='gray')
            ax[i,j].axis('off')
        index+=1
    plt.tight_layout()
    plt.savefig(f'{save_path}/{data_set}_SA_{mode}.pdf', bbox_inches='tight')
