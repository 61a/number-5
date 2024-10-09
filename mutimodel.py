import torch
import torch.nn as nn
from model.resnet import resnet18,resnet34,resnet50
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
        ## self.modal1.fc = nn.Identity()
        # self.modal1 = WideResNet()
        # self.modal1.conv1 = nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1,
        #                        padding=1, bias=True)
        
        # self.modal1.fc = nn.Identity()

        # 模态 2
        self.modal2 = resnet18(num_classes)
        self.modal2.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # self.modal2.fc = nn.Identity()
        # self.modal2 = WideResNet()
        # self.modal2.conv1 = nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1,
        #                        padding=1, bias=True)
        
        # self.modal2.fc = nn.Identity()

        # 模态 3
        self.modal3 = resnet18(num_classes)
        self.modal3.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # self.modal3 = WideResNet()
        # self.modal3.conv1 = nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1,
        #                        padding=1, bias=True)
        # self.modal3.fc = nn.Identity()

        # 跨模态融合
        # self.fc_fusion = nn.Conv2d(1280*2, num_classes, 1)
        self.fc_fusion = nn.Linear(3*2, num_classes)
        
        self.model4 = VAE(self.in_channels*3,self.latent_dim)
        
        self.linear = nn.Sequential(
            nn.Linear(16384, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Linear(512, num_classes), 
            nn.ReLU()
        )
        
        self.conv_fusion = nn.Sequential(  # resnet:512*3，512 # wideresnet 1152
            nn.Conv2d(512*3, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1) # wideresnet添加的
            )
        
    def beginModel(self, x):
        x = x.reshape(-1,3,32,32)
        # print(x.shape)
        o1,x1 = self.modal1(x[:,0,:,:].reshape(-1,1,32,32))  #o:最终输出 x:卷积特征输出
        o2,x2 = self.modal2(x[:,1,:,:].reshape(-1,1,32,32))
        o3,x3 = self.modal3(x[:,2,:,:].reshape(-1,1,32,32))
        
        # 特征级融合
        xi = torch.cat((x1, x2, x3), dim=1) # shape: torch.Size([32, 1536, 4, 4])
        xi = self.conv_fusion(xi)  # torch.Size([32, 512, 4, 4])
        
        o = torch.cat((o1, o2, o3), dim=1)
        
        

        # 全连接层
        o = self.fc_fusion(o)
        o = o.view(o.size(0), -1)

        return o,xi
    
    def VAEModel(self,x):
        
        x = x.reshape(-1,3,32,32)
        output = self.model4(x)   # output = [self.decode(z), input, mu, log_var]
        
        # result = self.model4.encoder(x)
        
        mu, log_var = self.model4.encode(x)
        z = self.model4.reparameterize(mu, log_var)
        result = self.model4.decoder_input(z)
        result = result.view(-1, 512, 4, 4) # resnet:(-1, 512, 4, 4)
        
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