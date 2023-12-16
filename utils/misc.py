import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
import os
import os.path
import torch.nn.functional as F
import torchvision.transforms as transforms
import shutil

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class TransformTwice:
    def __init__(self, transform, orig_out=False):
        self.transform = transform
        self.orig_out = orig_out

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)

        if self.orig_out:
            # PIL to tensor transform
            to_tensor = transforms.Compose([
                transforms.ToTensor()
            ])
            # Convert the image to Torch tensor
            orig = to_tensor(inp)
            return out1, out2, orig
        else:
            return out1, out2

class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target):
    
    num_correct = np.sum(output == target)
    res = num_correct / len(target)

    return res

def cluster_acc(y_pred, y_true):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    w = w[y_true.min():, y_true.min():]
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / y_pred.size

def entropy(x):
    """ 
    Helper function to compute the entropy over the batch 
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    EPS = 1e-5
    x_ =  torch.clamp(x, min = EPS)
    b =  x_ * torch.log(x_)

    if len(b.size()) == 2: # Sample-wise entropy
        return - b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))

class MarginLoss(nn.Module):
    
    def __init__(self, m=0.2, weight=None, s=10):
        super(MarginLoss, self).__init__()
        self.m = m
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        x_m = x - self.m * self.s
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target, weight=self.weight)

def degrade_data(degrade_level, degrade_choice):
    degrade_levels = [1.0, 2.0, 3.0]
    
    degrade = {'blur': transforms.GaussianBlur(kernel_size=(3,3), sigma=degrade_levels[degrade_level-1]),
                'jitter': transforms.ColorJitter(brightness=.2*int(degrade_level), hue=0.15*int(degrade_level)),
                'elastic': transforms.ElasticTransform(alpha=50.0*degrade_level)}
    return degrade[degrade_choice]

class Regularizer():

    def __init__(self, model):
        self.model = model

    # Function to get variance
    def get_var(self, x, device, num_classes, n_MC=10):
        x = x.to(device)
        
        softmax_results = torch.empty((len(x),0,num_classes)).to(device)
        # For each data point in the batch, we take n_MC samples
        for j in range(n_MC):
            output, _ = self.model(x)
            softmax_out = F.softmax(output, dim=1)
            softmax_results = torch.cat((softmax_results, softmax_out.unsqueeze(dim=1)), dim=1)
        softmax_mean = torch.mean(softmax_results, dim=1)
        # Return variance 
        softmax_var = softmax_results-torch.cat([softmax_mean.unsqueeze(dim=1)]*n_MC, dim=1)
        softmax_mm = torch.bmm(softmax_var, softmax_var.transpose(1,2)).diagonal(dim1=1,dim2=2)
        var = torch.mean(softmax_mm, dim=1)

        return var

    # The monotonic function taking in variance
    def monotonic_fn(post_var):
        return torch.subtract(1, torch.div(1, torch.add(1, post_var)))

    # Posterior regularizer with z=g(post_var)
    def post_regularizer(w, labeled_len):
        entropy = - torch.mean(w,0)*torch.log(torch.mean(w, 0)) - (1-torch.mean(w,0))*torch.log(1-torch.mean(w,0))
        return entropy

def MMD(z_s, z_t, kernel, device):
    """Emprical maximum mean discrepancy.

    Args:
        z_s: samples from source distribution P
        z_t: samples from target distibution P'
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(z_s, z_s.t()), torch.mm(z_t, z_t.t()), torch.mm(z_s, z_t.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.sum(XX + YY - 2. * XY)

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)