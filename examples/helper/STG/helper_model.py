import torch 
import torch.nn as nn

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
    
def getcolormodel():
    rgbdecoder = Sandwich(9,3)
    return rgbdecoder

def trbfunction(x): 
    return torch.exp(-1*x.pow(2))