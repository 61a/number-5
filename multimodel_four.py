import torch
import torch.nn as nn
from model.senet import seresnet18
from model.resnet import resnet18,resnet34,resnet50
from model.wrn import WideResNet
from model.mobilenetv2 import MobileNetV2
# from VAE.beta_vae import BetaVAE as VAE
from VAE.vae import BetaVAE as VAE
import numpy as np
import torch.nn.functional as F

class Net1D(nn.Module): #把输出看作feature
    def __init__(self,input_dim, hidden_dim, out_size, num, dropout):
        super(Net1D,self).__init__()
        self.num = num
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, stride=1)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=10, kernel_size=3, stride=1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.num, 512)
        self.fc2 = nn.Linear(512, out_size)
        # self.fc3 = nn.Linear(out_size,2)

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
    

class MultiModalModel(nn.Module):
    def __init__(self,args):
        super(MultiModalModel, self).__init__()
        
        self.in_channels = args.in_channels
        self.latent_dim = args.latent_dim
        self.n_classes = args.final_out


        # 模态 1
        self.modal1 = Net1D(input_dim=args.profile_in, hidden_dim=args.profile_hidden, out_size=args.profile_out, num= args.profile_linear, dropout=0.5)

        # 模态 2
        self.modal2 = Net1D(input_dim=args.DM_in, hidden_dim=args.DM_hidden, out_size=args.DM_out, num= args.DM_linear, dropout=0.5)

        # 模态 3
        # self.modal3 = resnet18(n_class=args.subband_out)
        # self.modal3.conv1 = nn.Sequential(
        #     nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True))
        
        # self.modal3 = MobileNetV2(class_num=args.subband_out)
        # self.modal3.pre = nn.Sequential(
        #     nn.Conv2d(self.in_channels, 32, 1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU6(inplace=True)
        # )
        
        self.modal3 = seresnet18(n_class=args.subband_out)
        self.modal3.pre = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 模态 4
        # self.modal4 = resnet18(n_class=args.subint_out)
        # self.modal4.conv1 = nn.Sequential(
        #     nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True))
        
        # self.modal4 = MobileNetV2(class_num=args.subint_out)
        # self.modal4.pre = nn.Sequential(
        #     nn.Conv2d(self.in_channels, 32, 1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU6(inplace=True)
        # )
        
        self.modal4 = seresnet18(n_class=args.subint_out)
        self.modal4.pre = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        
        
        self.model5_task1 = nn.Sequential(
            nn.Linear(8, 8),
            nn.Linear(8, self.n_classes)
        )
        
        self.model5_task2 = nn.Sequential(
            nn.Linear(8, 8),
            nn.Linear(8, self.n_classes)
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
        
        self.out = self.beginModel(profile,DM,subband,subint)
        if task == 'task1':
            final = self.model5_task1(self.out)
        elif task == 'task2':
            final = self.model5_task2(self.out)
        else:
            raise ValueError("Unknown task")
        # final = self.model5(self.out)   # 
        # pre = self.model5.classifier(out)  
        return final 
    
    # def update_for_new_task(self, new_classes):
    #     # 更新类别总数
    #     self.n_classes += new_classes
    #     # 更新模型以适应新任务
    #     self.model5 = nn.Linear(8, self.n_classes)



class build_multimodel18:
    def __init__(self, is_remix=False):
        self.is_remix = is_remix

    def build(self, num_classes):
        return MultiModalModel(num_classes=num_classes)