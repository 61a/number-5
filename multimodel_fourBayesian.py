import torch
import torch.nn as nn
from model.BayesianSenet import seresnet18
import numpy as np
import torch.nn.functional as F
from util.BBBConv import BBBConv2d,BBBConv1d,BBBLinear

class Net1D(nn.Module): #把输出看作feature
    def __init__(self,input_dim, hidden_dim, out_size, num, dropout):
        super(Net1D,self).__init__()
        self.num = num
        self.conv1 = BBBConv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, stride=1)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = BBBConv1d(in_channels=hidden_dim, out_channels=10, kernel_size=3, stride=1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = BBBLinear(self.num, 512)
        self.fc2 = BBBLinear(512, out_size)

    def forward(self,x):

        x = x.unsqueeze(1) 
        out = F.relu(self.conv1(x))
        out = self.norm1(out)
        out = F.relu(self.conv2(out))
        out = out.view(-1,self.num)
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out
    
    def get_kl(self):
        kl = 0.0
        kl += self.conv1.kl_loss()
        kl += self.conv2.kl_loss()
        kl += self.fc1.kl_loss()
        kl += self.fc2.kl_loss()
        return kl
    
    def update_prior(self):
        # 针对两个卷积层进行更新
        for module in [self.conv1, self.conv2]:
            module.prior_W_mu = module.W_mu.data.clone()
            module.W_sigma = torch.log1p(torch.exp(module.W_rho))  # 计算 W_sigma
            module.prior_W_sigma = module.W_sigma.data.clone()
            if module.use_bias:
                module.prior_bias_mu = module.bias_mu.data.clone()
                module.bias_sigma = torch.log1p(torch.exp(module.bias_rho))  # 如果使用了bias, 也要计算 bias_sigma
                module.prior_bias_sigma = module.bias_sigma.data.clone()

        # 针对两个全连接层进行更新
        for module in [self.fc1, self.fc2]:
            module.prior_W_mu = module.W_mu.data.clone()
            module.W_sigma = torch.log1p(torch.exp(module.W_rho))  # 计算 W_sigma
            module.prior_W_sigma = module.W_sigma.data.clone()
            if module.use_bias:
                module.prior_bias_mu = module.bias_mu.data.clone()
                module.bias_sigma = torch.log1p(torch.exp(module.bias_rho))  # 如果使用了bias, 也要计算 bias_sigma
                module.prior_bias_sigma = module.bias_sigma.data.clone()
    
    

class MultiModalModel(nn.Module):
    def __init__(self,args):
        super(MultiModalModel, self).__init__()
        
        self.in_channels = args.in_channels
        self.latent_dim = args.latent_dim

        # 模态 1
        self.modal1 = Net1D(input_dim=args.profile_in, hidden_dim=args.profile_hidden, out_size=args.profile_out, num= args.profile_linear, dropout=0.5)

        # 模态 2
        self.modal2 = Net1D(input_dim=args.DM_in, hidden_dim=args.DM_hidden, out_size=args.DM_out, num= args.DM_linear, dropout=0.5)

        
        self.modal3 = seresnet18(n_class=args.subband_out)
        self.modal3.pre = nn.Sequential(
            BBBConv2d(self.in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        
        self.modal4 = seresnet18(n_class=args.subint_out)
        self.modal4.pre = nn.Sequential(
            BBBConv2d(self.in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

                
        self.model5_task1 = nn.Sequential(
            BBBLinear(8, 8),
            BBBLinear(8, args.final_out)
        )
        
        self.model5_task2 = nn.Sequential(
            BBBLinear(8, 8),
            BBBLinear(8, args.final_out)
        )
        
        
    def beginModel(self, profile,DM,subband,subint):

        o1 = self.modal1(profile)  #o:最终输出 
        o2 = self.modal2(DM)
        o3 = self.modal3(subband)
        o4 = self.modal4(subint)
        
        # 特征级融合
        x = torch.cat((o1, o2, o3, o4), dim=1)  

        return x
    

    def forward(self, profile,DM,subband,subint,task):
        self.task = task
        self.out = self.beginModel(profile,DM,subband,subint)
        if self.task == 'task1':
            final = self.model5_task1(self.out)
        elif self.task == 'task2':
            final = self.model5_task2(self.out)
        else:
            raise ValueError("Unknown task") 
        return final 
    
    
    def get_kl(self):
        kl = 0.0
        # Accumulate KL divergence from all modal components
        for modal in [self.modal1, self.modal2, self.modal3, self.modal4]:
            kl += modal.get_kl()
        # Additionally, iterate over the model5 to accumulate its KL divergence 
        if self.task == 'task1':     
            for layer in self.model5_task1:
                if hasattr(layer, 'kl_loss'):
                    kl += layer.kl_loss()
        elif self.task == 'task2':     
            for layer in self.model5_task2:
                if hasattr(layer, 'kl_loss'):
                    kl += layer.kl_loss()
        return kl
    
    def update_prior(self):
        # Update priors for all modal components
        for modal in [self.modal1, self.modal2, self.modal3, self.modal4]:
            modal.update_prior()
        # Additionally, update priors for the model5 if they are BBB layers
        for layer in self.model5_task1:
            if isinstance(layer, BBBLinear):  # Check if the layer is a BBBLinear layer
                # Update weight priors
                layer.prior_W_mu = layer.W_mu.data.clone()
                layer.W_sigma = torch.log1p(torch.exp(layer.W_rho))  # 计算 W_sigma
                layer.prior_W_sigma = layer.W_sigma.data.clone()
                # Update bias priors if bias is used
                if layer.use_bias:
                    layer.prior_bias_mu = layer.bias_mu.data.clone()
                    layer.bias_sigma = torch.log1p(torch.exp(layer.bias_rho))  # 如果使用了bias, 也要计算 bias_sigma
                    layer.prior_bias_sigma = layer.bias_sigma.data.clone()




class build_multimodel18:
    def __init__(self, is_remix=False):
        self.is_remix = is_remix

    def build(self, num_classes):
        return MultiModalModel(num_classes=num_classes)