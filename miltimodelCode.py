import torch
import torch.nn as nn
from model.resnet import resnet18
from model.wrn import WideResNet
from model.mobilenetv2 import MobileNetV2
from VAE.beta_vae import BetaVAE as VAE
import numpy as np


class MultiModalModel(nn.Module):
    def __init__(self, num_classes,args):
        super(MultiModalModel, self).__init__()
        
        self.in_channels = args.in_channels
        self.latent_dim = args.latent_dim

        # 模态 1
        self.modal1 = VAE(self.in_channels,self.latent_dim)

        # 模态 2
        self.modal2 = VAE(self.in_channels,self.latent_dim)

        # 模态 3
        self.modal3 = VAE(self.in_channels,self.latent_dim)

        
        
        self.linear = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64), 
            nn.ReLU(),
            nn.Linear(64, num_classes), 
            nn.ReLU()
        )
        
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(512*3, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU())
        
    def beginModel(self, x):
        x = x.reshape(-1,3,32,32)
        # print(x.shape)
        o1 = self.modal1(x[:,0,:,:].reshape(-1,1,32,32))  # [result, input, mu, log_var, self.result]
        o2 = self.modal2(x[:,1,:,:].reshape(-1,1,32,32))
        o3 = self.modal3(x[:,2,:,:].reshape(-1,1,32,32))
        
        # 特征级融合
        xi = torch.cat((o1[-1], o2[-1], o3[-1]), dim=1) # shape: torch.Size([32, 512*3, 4, 4])
        xi = self.conv_fusion(xi)  # torch.Size([32, 512, 4, 4])
        
        xi = xi.view(xi.size(0), -1)
        
        o = self.linear(xi)

        return o1,o2,o3,o
    

    def forward(self, x):
        
        o1,o2,o3,o = self.beginModel(x)

        
        
        return [o1,o2,o3,o]
    
    # def loss(self):
    #     loss = 0
    #     return loss
