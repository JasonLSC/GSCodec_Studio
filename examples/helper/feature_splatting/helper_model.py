import torch 
import torch.nn as nn
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
import math

class Sandwich_static(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(Sandwich_static, self).__init__()
        
        self.mlp1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias) # 
        self.mlp2 = nn.Conv2d(dim, 3, kernel_size=1, bias=bias)
        # self.mlp1 = nn.Conv2d(dim, 6, kernel_size=1, bias=bias) # 
        # self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()

        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, input, rays, time=None):
        # albedo, spec, timefeature = input.chunk(3,dim=1)
        # specular = torch.cat([spec, timefeature, rays], dim=1) # 3+3 + 5
        # albedo, spec = input.chunk(2,dim=1)
        albedo = input[:,0:3]
        spec = input[:,3:]
        specular = torch.cat([spec, rays], dim=1)
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)
        
        result = albedo + 0.1 * specular
        # result = albedo + specular
        result = self.sigmoid(result) 
        return result

def getcolormodel_static(dim = 12):
    rgbdecoder = Sandwich_static(dim,3)
    return rgbdecoder

def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0