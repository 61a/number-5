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
        self.modal1 = resnet18(num_classes)
        self.modal1.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        
        self.model4 = VAE(self.in_channels,self.latent_dim)
        
        self.linear = nn.Sequential(
            nn.Linear(16384, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512), 
            nn.ReLU(),
            nn.Linear(512, 128), 
            nn.ReLU(),
            nn.Linear(128, num_classes), 
            nn.ReLU()
        )
        
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        
    def beginModel(self, x):
        x = x.reshape(-1,3,32,32)
        # print(x.shape)
        o,xi = self.modal1(x)  #o:最终输出 x:卷积特征输出

        xi = self.conv_fusion(xi)  # torch.Size([32, 512, 4, 4])

        return o,xi
    
    def VAEModel(self,x):
        
        x = x.reshape(-1,3,32,32)
        output = self.model4(x)   # output = [self.decode(z), input, mu, log_var]
        
        mu, log_var = self.model4.encode(x)
        z = self.model4.reparameterize(mu, log_var)
        result = self.model4.decoder_input(z)
        result = result.view(-1, 512, 4, 4)
        
        # result = self.model4.encoder(x)
        
        return output, result

    def forward(self, x):
        
        pred,output1 = self.beginModel(x)
        recons, output2 = self.VAEModel(x)

        output = torch.cat((output1,output2),dim=1)
        
        out = output.view(output.size(0), -1)
        out = self.linear(out)
        
        
        return [pred,output1,recons,output2,out]
    
    # def loss(self):
    #     loss = 0
    #     return loss


class build_multimodel18:
    def __init__(self, is_remix=False):
        self.is_remix = is_remix

    def build(self, num_classes):
        return MultiModalModel(num_classes=num_classes)