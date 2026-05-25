

  
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import math
import argparse
import os
from PIL import Image
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils
import yaml


def batch_save_tensors(tensor_list, output_dir="output_images",prefix='image_'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Starting save process for {len(tensor_list)} images...")
    for i, tensor in enumerate(tensor_list, start=1):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
            
        file_path = os.path.join(output_dir, f"{prefix}{i}.png")
        vutils.save_image(tensor, file_path)


def give_class_average_image(train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    class_images = [[] for _ in range(10)]
    for images, labels in train_loader:
        for i in range(10):
            class_images[i].append(images[labels == i])
    for i in range(10):
        class_images[i] = torch.cat(class_images[i], dim=0).mean(dim=0)
    return class_images

def class_average_image(dataset):
    if dataset == 'mnist':
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    elif dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    elif dataset == 'fashion-mnist':
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    elif dataset == 'k-mnist':
        train_dataset = datasets.KMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return give_class_average_image(train_dataset)



def save_check_point(state_dict, path, save_name):
    """
    Save a model checkpoint.

    Args:
        state_dict (dict): State dictionary of the model/optimizer.
        path (str): Directory path to save the checkpoint.
        save_name (str): Name of the saved checkpoint file.

    Returns:
        None
    """
    os.makedirs(path, exist_ok=True)
    torch.save(state_dict, path)


def get_config_path(path):

    """
    Load YAML config file and convert it into a namespace.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        argparse.Namespace: Namespace object with config parameters.
    """


    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
        config = dict2namespace(config)
    return config


# Function to simulate GBM at a single time point


def gen_x_t(x_0, t, mu, sigma, dt=1e-3):
    """
    Simulate one step of Geometric Brownian Motion (GBM).

    Args:
        x_0 (torch.Tensor): Initial value(s).
        t (int): Time step.
        mu (float): Drift coefficient.
        sigma (float): Volatility.
        dt (float, optional): Time step size. Default = 1e-3.

    Returns:
        torch.Tensor: Value at time t.
    """
    return x_0 * torch.exp((mu - 0.5 * sigma**2) * t * dt + sigma * torch.sqrt(t * dt) * torch.randn_like(x_0))


# Load all images onto the device at once
def load_all_images(loader, device):
    """
    Load all images from a DataLoader into a single tensor.

    Args:
        loader (DataLoader): Torch DataLoader.
        device (str/torch.device): Device to store the images.

    Returns:
        torch.Tensor: Concatenated images [N, C, H, W].
    """

    all_images = []
    for images, _ in loader:
        images = images.to(device)
        all_images.append(images)
    return torch.cat(all_images, dim=0)



def dict2namespace(config):
    """
    Recursively convert a nested dictionary into an argparse.Namespace.

    Args:
        config (dict): Dictionary of configuration values.

    Returns:
        argparse.Namespace
    """

    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def generate_spirals(n_samples, spacing_factor=0.5):
    u1 = np.random.uniform(low=0, high=1, size=(n_samples // 2))
    u2 = np.random.uniform(low=0, high=1, size=(n_samples // 2))
    u3 = np.random.uniform(low=0, high=1, size=(n_samples // 2))

    n = np.sqrt(u1) * 540 * (2 * np.pi) / 360 * spacing_factor
    d1x = -np.cos(n) * n + u2 * 0.5
    d1y = np.sin(n) * n + u3 * 0.5
    x = np.concatenate([np.stack([d1x, d1y], axis=1),
                        np.stack([-d1x, -d1y], axis=1)], axis=0) / 3
    x += 5  # Shift to ensure all points are positive
    np.random.shuffle(x)
    return x.astype(np.float32)


def plot_samples():
    # Parameters for GBM
    sigma = 0.2  # Volatility
    mu = 2.5 * sigma  # Drift coefficient
    T = 1.0  # Time horizon
    N = 5000  # Number of time steps
    M = 1000  # Number of sample paths

    # Time discretization
    dt = T / N
    t = np.linspace(0, T, N)

    # Parameters
    n_samples = 1000

    # Generate initial samples from the two spirals pattern
    spirals = generate_spirals(n_samples)
    initial_samples = spirals[:M]  # Select M samples

    # Simulate 2-dimensional GBM paths
    X = np.zeros((M, N, 2))
    X[:, 0, :] = initial_samples  # Initial values from the spirals pattern
    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt), (M, 2))
        X[:, i, :] = X[:, i-1, :] * \
            np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

    # Plot the initial samples and GBM paths
    plt.figure(figsize=(20, 6))

    # Plot the initial samples
    plt.subplot(1, 3, 1)
    plt.scatter(initial_samples[:, 0],
                initial_samples[:, 1], c='blue', alpha=0.6)
    plt.title('Initial Samples (Spirals)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    # Plot the GBM paths
    plt.subplot(1, 3, 2)
    for i in range(M):
        plt.plot(X[i, :, 0], X[i, :, 1], lw=0.5, alpha=0.7)
    plt.scatter(X[:, 0, 0], X[:, 0, 1], c='red', alpha=0.6, label='Start')
    plt.scatter(X[:, -1, 0], X[:, -1, 1], c='green', alpha=0.6, label='End')
    plt.title('2D GBM Paths')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)

    # Calculate the 2D histogram of the points at the end of the GBM paths
    end_points = X[:, -1, :]
    hist, x_edges, y_edges = np.histogram2d(
        end_points[:, 0], end_points[:, 1], bins=200)

    # Plot the 2D histogram
    plt.subplot(1, 3, 3)
    plt.imshow(hist.T, origin='lower', extent=[
               x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect='auto', cmap='viridis')
    plt.colorbar(label='Frequency')
    plt.title('2D Histogram of End Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("./assets/2d_gbm_paths.png")
    plt.close()

    # Fit a 2D log-normal distribution to the end points
    shape1, loc1, scale1 = lognorm.fit(end_points[:, 0], floc=0)
    shape2, loc2, scale2 = lognorm.fit(end_points[:, 1], floc=0)

    # Generate the log-normal PDF for each dimension
    x1 = np.linspace(np.min(end_points[:, 0]), np.max(end_points[:, 0]), 1000)
    x2 = np.linspace(np.min(end_points[:, 1]), np.max(end_points[:, 1]), 1000)
    pdf1 = lognorm.pdf(x1, shape1, loc=loc1, scale=scale1)
    pdf2 = lognorm.pdf(x2, shape2, loc=loc2, scale=scale2)

    # Plot the fitted 2D log-normal density
    plt.figure(figsize=(8, 6))
    X1, X2 = np.meshgrid(x1, x2)
    Z = lognorm.pdf(X1, shape1, loc=loc1, scale=scale1) * \
        lognorm.pdf(X2, shape2, loc=loc2, scale=scale2)
    plt.contourf(X1, X2, Z, cmap='viridis')
    plt.colorbar(label='Density')
    plt.title('Fitted 2D Log-Normal Density')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.savefig("./assets/2d_lognormal_density.png")
    plt.close()

    parameters = {
        "dim1": {"shape": shape1, "loc": loc1, "scale": scale1},
        "dim2": {"shape": shape2, "loc": loc2, "scale": scale2}
    }

    print(parameters)

    return None


def get_lognormal_params(mean, var, skew, kurt):
    """
    Convert moments to lognormal parameters using the correct skewness formula.
    For a lognormal distribution:
    skewness = (exp(σ²) + 2) * sqrt(exp(σ²) - 1)
    where σ is the standard deviation of the underlying normal distribution
    """
    # Solve for σ using skewness formula
    def skew_equation(sigma):
        exp_sigma2 = np.exp(sigma**2)
        return (exp_sigma2 + 2) * np.sqrt(exp_sigma2 - 1) - skew

    # Use numerical solver to find σ
    from scipy.optimize import fsolve
    sigma = float(fsolve(skew_equation, x0=1.0)[0])

    # Calculate μ using the mean formula:
    # mean = exp(μ + σ²/2)
    mu = np.log(mean) - sigma**2/2

    return mu, sigma


def sample_lognormal(mean, var, skew, kurt, size, device='cuda'):
    """Draw samples from a lognormal with specified moments"""
    mu, sigma = get_lognormal_params(mean, var, skew, kurt)

    # Generate samples using torch
    normal_samples = torch.randn(size, device=device) * sigma + mu
    lognormal_samples = torch.exp(normal_samples)

    return lognormal_samples



class CelebADataset(VisionDataset):
    def __init__(self, root, transform=None):
        super(CelebADataset, self).__init__(root, transform=transform)
        self.images = [os.path.join(root, img) for img in os.listdir(
            root) if img.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, 0  # Return a dummy label


def show_grid(images, title, step, dataset='mnist', save_path=None, gen=False):
    images = images.cpu().numpy()
    num_images = int(math.sqrt(images.shape[0]))
    # Display a grid of 8x8 images
    fig, axes = plt.subplots(num_images, num_images, figsize=(12, 12))

    # Check if images are RGB (3 channels) or grayscale (1 channel)
    is_rgb = images.shape[1] == 3

    for i in range(num_images):
        for j in range(num_images):
            ax = axes[i, j]
            if is_rgb:
                # For RGB images, transpose to (H,W,C) format and clip to valid range
                img = np.transpose(images[i * num_images + j], (1, 2, 0))
                img = np.clip(img, 0, 1)
                ax.imshow(img)
            else:
                # For grayscale images
                ax.imshow(images[i * num_images + j].squeeze(), cmap='gray')
            ax.axis('off')

    plt.suptitle(f"{title} - Step {step}")
    filename = title.replace(' ', '_')
    # path change mnist_exps/images/mnist/gen
    if save_path is None:
        plt.savefig(f"mnist_exps/images/{dataset}/gen/{filename}_step={step}.png") if gen else plt.savefig(
            f"mnist_exps/images/{dataset}/test/{filename}_step={step}.png")
        plt.close()
    else:
        plt.savefig(f"{save_path}/gen/{filename}_step={step}.png") if gen else plt.savefig(
            f"{save_path}/test/{filename}_step={step}.png")
        plt.close()


def forward_sde_gbm(mu, sigmas, x_now, N, T, device='cpu'):

    """
    Forward SDE simulation using GBM with varying sigmas.

    Args:
        mu (torch.Tensor): Drift coefficient.
        sigmas (torch.Tensor): Array of volatilities for each step.
        x_now (torch.Tensor): Initial input images [B, C, H, W].
        N (int): Number of steps.
        T (float): Time horizon.
        device (str): Device.

    Returns:
        torch.Tensor: Simulated samples [B, N, C, H, W].
    """


    dt = T / N
    samples = torch.zeros(
        (x_now.shape[0], N, x_now.shape[1], x_now.shape[2], x_now.shape[3])).to(device)
    for i in range(N):
        sigma = sigmas[i].to(device)
        x_now = x_now * torch.exp((mu - 0.5 * sigma**2) * dt +
                                  sigma * math.sqrt(dt) * torch.randn_like(x_now).to(device))
        samples[:, i, :, :, :] = x_now
    return samples


def forward_sde_gbm_fast(mu, sigma, x_now, t, N, T=1, device='cpu'):

    """
    Fast GBM sampling at a single time t.

    Args:
        mu (torch.Tensor): Drift.
        sigma (torch.Tensor): Volatility.
        x_now (torch.Tensor): Initial images.
        t (torch.Tensor): Current step index.
        N (int): Total steps.
        T (float): Time horizon.
        device (str): Device.

    Returns:
        torch.Tensor: Noised images at time t.

    """

    t = t.view(-1, 1, 1, 1)
    sigma = sigma.view(-1, 1, 1, 1)
    mu = mu.view(-1, 1, 1, 1)
    dt = T / N
    x_t = x_now * torch.exp((mu - 0.5 * sigma**2) * dt * t + sigma *
                            torch.sqrt(dt * t) * torch.randn_like(x_now).to(device))
    return x_t


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# Plot and save histogram of pixel intensities
def plot_histogram(images, title, dataset='mnist', path=None):
    images = images.cpu().numpy().flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(images, bins=50, color='black', alpha=0.7, density=True)

    # Fit a lognormal distribution to the data
    shape, loc, scale = lognorm.fit(images, floc=0)
    x = np.linspace(images.min(), images.max(), 1000)
    pdf = lognorm.pdf(x, shape, loc, scale)
    mean, var, skew, kurt = lognorm.stats(shape, loc, scale, moments='mvsk')

    # Plot the fitted lognormal density
    plt.plot(x, pdf, 'r-', lw=2, label='Lognormal fit')

    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.legend()
    filename = title.replace(' ', '_')
    if path is None:
        os.makedirs(f"mnist_exps/images/{dataset}/test", exist_ok=True)
        plt.savefig(
            f"mnist_exps/images/{dataset}/test/{filename}_histogram.png")
    else:
        plt.savefig(f"{path}/test/{filename}_histogram.png")
    plt.close()
    return mean, var, skew, kurt


def test_forward(sigma, N, path=None, device='cpu'):
    if path is not None:
        os.makedirs(path+'/gen', exist_ok=True)
        os.makedirs(path+'/test', exist_ok=True)
    N = N
    mu = torch.tensor([0.0]).to(device=device)
    # sigmas = torch.linspace(0.01, 1, N).to(device)  # Increase the range of sigmas for more aggressive noise
    # sigmas = torch.logspace(-1, 0.5, N).to(device)  # Increase the range of sigmas for more aggressive noise
    sigmas = sigma * torch.ones(N).to(device)
    delta = 1 / N

    # Define transformations for the datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST dataset
    mnist_train = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    batch_size = 64
    mnist_train_loader = DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True)
    mnist_test_loader = DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False)

    images = next(iter(mnist_train_loader))[0].to(device) + 1.0

    show_grid(images.detach().cpu(), title="Original Images", step=0)

    for i in range(N):
        x_now = forward_sde_gbm_fast(mu, sigmas[i], images, torch.tensor(
            i).to(device), N, T=1, device=device)

        if (i+1) % 10 == 0 or i == N-1:
            print(f"Step {i+1}: {x_now.min()}, {x_now.max()}")
            show_grid((x_now-1).detach().cpu(),
                      title=f"Noisy Images at step {i+1}", step=i+1, save_path=path, gen=True)
            plot_histogram(
                x_now, title=f"Histogram at step {i+1}", dataset='cifar10', path=path)

    # Plot pure noise
    noise = torch.randn_like(images).to(device)
    log_normal_noise = torch.exp(
        (mu - (sigmas[-1]**2)/2) * (i * delta) + math.sqrt(i * delta) * noise)
    show_grid(log_normal_noise.detach().cpu(), title="Lognormal Noise", step=N)
    print(
        f"Lognormal Noise: {log_normal_noise.min()}, {log_normal_noise.max()}")
    plot_histogram(log_normal_noise, title=f"Histogram of noise",
                   dataset='cifar10', path=path)

    return noise


class K_Mnist(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path
        self.transform = transform if transform != None else transforms.Compose([
            transforms.ToTensor()
        ])
        self.x_images = torch.from_numpy(
            np.load(path)['arr_0']).unsqueeze(dim=1)

    def __len__(self):
        return self.x_images.shape[0]

    def __getitem__(self, index):

        images = self.x_images[index]
        return images/images.max()


# Load all images onto the device at once


def load_all_images(loader, device):
    all_images = []
    for images, _ in loader:
        images = images.to(device)
        all_images.append(images)
    return torch.cat(all_images, dim=0)

from tqdm._tqdm import tqdm
def save_grid_pdf(tensors,rows,cols,save_name):
    """
    Save a grid of tensors as a PDF.

    Args:
        tensors (torch.Tensor): Images [N, C, H, W].
        rows (int): Rows in grid.
        cols (int): Columns in grid.
        save_name (str): Output PDF filename.

    Returns:
        None
    """
    index = 0
    total = tensors.shape[0]
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(rows,cols))
    for i in tqdm(range(rows)):
        for j in range(cols):
            ax[i,j].imshow(tensors[index].permute(1, 2, 0).numpy(),
                        aspect='auto', cmap='gray')
            ax[i,j].axis('off')
            ax[i,j].set_aspect('equal')
            ax[i,j].set_xticklabels([])
            ax[i,j].set_yticklabels([])
            index+=1
    fig.tight_layout(pad=0.0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(save_name+".pdf")
    

def save_images_from_tensors(tensor_out,save_path,start_num=0):
    for i in range(len(tensor_out)):
        image = tensor_out[i]* 255
        #save image 
        image = Image.fromarray((image).astype(np.uint8))
        image.save(f"{save_path}/sample_{start_num + i}.png")




def get_config_path(path):
    """
    Loads a YAML configuration file from the specified path and converts it into a namespace-like object.

    Parameters:
        path (str): 
            The file path to the YAML configuration file.

    Returns:
        Namespace:
            A namespace-like object (typically implemented using a custom `dict2namespace` function) 
            that allows access to configuration keys using dot notation (e.g., `config.learning_rate`).

    Notes:
        - This function depends on the `yaml` module for parsing and a `dict2namespace` function 
          (defined in utils.py) that converts dictionaries to objects.
    """
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
        config = dict2namespace(config)
    return config

def forward_sde_gbm_channelwise(mu, sigma, x_now, t, N, T=1, device='cpu'):
    """
    Apply GBM noise channel-wise without broadcasting.

    mu and sigma should be per-channel (e.g. shape [C,1,1,1] or [C]).
    x_now: [B,C,H,W]
    t: scalar or tensor of shape [B]
    Returns: x_t of same shape as x_now
    """
    x_now = x_now.to(device)
    dtype = x_now.dtype
    B, C, H, W = x_now.shape

    dt = T / N

    # t may be a Python scalar, a 0-d tensor, or a 1-D tensor with a single element
    # e.g., torch.tensor([999]) will be accepted and treated as scalar
    if isinstance(t, (int, float)):
        t_scalar = float(t)
    elif torch.is_tensor(t):
        if t.dim() == 0:
            t_scalar = float(t.item())
        elif t.dim() == 1 and t.numel() == 1:
            t_scalar = float(t.view(-1)[0])
        else:
            raise ValueError("t must be a scalar (0-d) or a 1-D tensor with a single element for channel-wise GBM")
    else:
        raise ValueError("t must be a scalar (0-d) or a 1-D tensor with a single element for channel-wise GBM")

    # prepare mu/sigma as channel vectors [C]
    mu_c = mu.to(device=device, dtype=dtype)
    sigma_c = sigma.to(device=device, dtype=dtype)

    def _to_C_vector(p):
        # convert p to a 1-D tensor of length C (per-channel values)
        if torch.is_tensor(p):
            if p.dim() == 4 and p.shape[0] == C and p.shape[1] == 1:
                return p.view(C)
            if p.dim() == 4 and p.shape[0] == 1 and p.shape[1] == C:
                return p.view(C)
            if p.dim() == 1 and p.shape[0] == C:
                return p
            if p.numel() == C:
                return p.view(C)
            if p.numel() == 1:
                return p.view(1).expand(C)
        # if not a tensor, try to make tensor
        p_t = torch.tensor(p, device=device, dtype=dtype)
        if p_t.numel() == C:
            return p_t.view(C)
        return p_t.view(1).expand(C)

    mu_c = _to_C_vector(mu_c)
    sigma_c = _to_C_vector(sigma_c)

    noise = torch.randn_like(x_now).to(device)
    x_t = torch.empty_like(x_now)

    # apply channel-wise (explicit loop so each channel uses its own mu/sigma)
    if t_scalar < 0:
        raise ValueError("t must be non-negative")

    for c in range(C):
        mu_ch = mu_c[c].to(device=device, dtype=dtype)
        sigma_ch = sigma_c[c].to(device=device, dtype=dtype)
        coeff = (mu_ch - 0.5 * sigma_ch ** 2) * (dt * t_scalar)
        stdev = sigma_ch * math.sqrt(dt * t_scalar)
        # noise[:, c, :, :] has shape [B, H, W]
        factor = torch.exp(coeff + stdev * noise[:, c, :, :])
        x_t[:, c, :, :] = x_now[:, c, :, :] * factor

    return x_t
