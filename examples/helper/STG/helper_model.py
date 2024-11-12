import torch 
import torch.nn as nn
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F

class Sandwich(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(Sandwich, self).__init__()
        
        self.mlp1 = nn.Conv2d(12, 6, kernel_size=1, bias=bias) # 

        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()

        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, input, rays, time=None):
        albedo, spec, timefeature = input.chunk(3,dim=1)
        specular = torch.cat([spec, timefeature, rays], dim=1) # 3+3 + 5
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)

        result = albedo + specular
        result = self.sigmoid(result) 
        return result

class Sandwich_static(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(Sandwich_static, self).__init__()
        
        self.mlp1 = nn.Conv2d(9, 6, kernel_size=1, bias=bias) # 

        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()

        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, input, time=None):
        # albedo, spec, timefeature = input.chunk(3,dim=1)
        # specular = torch.cat([spec, timefeature, rays], dim=1) # 3+3 + 5
        # albedo, spec = input.chunk(2,dim=1)
        albedo = input[:,0:3]
        spec = input[:,3:]
        specular = spec
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)

        result = albedo + specular
        result = self.sigmoid(result) 
        return result

def getcolormodel():
    rgbdecoder = Sandwich(9,3)
    return rgbdecoder

def getcolormodel_static():
    rgbdecoder = Sandwich_static(9,3)
    return rgbdecoder

def trbfunction(x): 
    return torch.exp(-1*x.pow(2))

def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)