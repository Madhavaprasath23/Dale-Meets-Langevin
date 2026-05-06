import torch.nn as nn
import torch
import torch.nn.functional as F


def gbm_loss(score, x_t, x_0, t, mu, sigma):
    """
    Fixed!
    """
    true_score = - (sigma ** 2) * (t ** 2) - torch.log(x_t *
                                                       torch.reciprocal(x_0)) + t * (mu - 0.5 * (sigma ** 2))
    loss = torch.mean(((sigma ** 2) * (t ** 2) * x_t * score - true_score)**2)
    return loss


def new_loss(k, sigma, xk, x0, mu, score, N=1000, reduction='mean'):
    out = score
    l_xk = torch.log(xk).to(out.device)
    l_x0 = torch.log(x0).to(out.device)
    kd = k/N
    kd = kd.view(-1, 1, 1, 1).to(out.device)
    s2 = torch.pow(sigma, 2).to(out.device)

    term1 = kd*s2 * xk * out

    term2 = l_xk - l_x0
    term3 = -mu*kd + 1.5 * s2 * (kd*torch.ones_like(xk))
    if reduction == 'sum':
        return torch.sum((term1 + term2 + term3) ** 2)
    elif reduction == 'mean':
        return torch.sum((term1 + term2 + term3) ** 2, dim=(1, 2, 3)).mean()
    else:
        raise Exception("The Loss either be Sum or mean")
