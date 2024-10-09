import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import scipy.io as scio
from scipy.io import loadmat
import torchvision
import os
import argparse
import numpy as np
import random
from data_four import MultiFeatureDataset, PulsarDataLoaderManager
from multimodel_fourBayesian import MultiModalModel
from loss import FocalLoss, EqualizedFocalLoss


import matplotlib.pyplot as plt
import pdb
# import transforms
# from dataset import CUB_200_2011_Train, CUB_200_2011_Test
import torchvision.transforms as tfs
import torchvision.datasets as datasets
from sklearn.metrics import *
import logging

class ELBO(nn.Module):

    def __init__(self, model, train_size, beta):
        super().__init__()
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.beta = beta
        self.train_size = train_size

    def forward(self, outputs, targets, kl):
        assert not targets.requires_grad
        return criterion(outputs.squeeze(),targets)  + self.beta * kl / self.num_params


def train(epoch):
    logger.info('Epoch: %d' % epoch)
    net.train()

    for batch_idx, data in enumerate(trainloader):  
                 
        profiles = data['cand_profile'].to(device)
        dm_curves = data['cand_dm_curve'].to(device)
        subbands = data['cand_subbands'].to(device)
        subints = data['cand_subints'].to(device)
        targets = data['label'].to(device).float()

        optimizer.zero_grad()
        outputs = net(profiles, dm_curves, subbands, subints,'task1')  # [pred,output1,recons,output2,out]

        kl = net.get_kl()
        loss = elbo(outputs, targets, kl)
     
        loss.backward()
        optimizer.step()
        net.zero_grad()

        

        if batch_idx % len(trainloader)*20 == 0:
            logger.info(f'loss: {loss:.6f}')
    

    return loss

def test(epoch):
    global best_eval_F1
    net.eval()
    y_true = []
    y_pred = []
    y_logits = []
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            profiles = data['cand_profile'].to(device)
            dm_curves = data['cand_dm_curve'].to(device)
            subbands = data['cand_subbands'].to(device)
            subints = data['cand_subints'].to(device)
            targets = data['label'].to(device)
           
            outputs = net(profiles, dm_curves, subbands, subints,'task1') #[pred,output1,recons,output2,out]

            # 这里使用sigmoid函数因为是二分类问题
            probs = torch.sigmoid(outputs.squeeze()).cpu()
            preds = (probs > 0.5).int()  # 使用0.5作为阈值
            
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(preds.tolist())
            y_logits.extend(probs.tolist())
            
        
        top1 = accuracy_score(y_true, y_pred)
        #top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        #top5 = 'HTRU none'
        precision = precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=1)
        recall = recall_score(y_true, y_pred, average='binary', pos_label=1)
        _F1 = f1_score(y_true, y_pred, average='binary',pos_label=1)
        # kappa = cohen_kappa_score(y_true, y_pred)
        #print(y_true)
        
        AUC = roc_auc_score(y_true, y_pred, multi_class='ovo')
        #AUC = 'HTRU none'

        cf_mat = confusion_matrix(y_true, y_pred, normalize=None)
        logger.info(f'confusion matrix:\n {np.array_str(cf_mat)}')
        
        # Save checkpoint.
        if _F1 > best_eval_F1:
            state = {
            'net': net.state_dict(),
            'F1': _F1,
            'epoch': epoch,
            }
            best_eval_F1 = _F1
            if not os.path.isdir('/aidata/Ly61/number5/CL0322/ckpt/{}/{}'.format(args.Dataset,args.version)):
                os.makedirs('/aidata/Ly61/number5/CL0322/ckpt/{}/{}'.format(args.Dataset,args.version))
            torch.save(state, '/aidata/Ly61/number5/CL0322/ckpt/{}/{}/ckpt_best.pth'.format(args.Dataset,args.version))
        # Save checkpoint.
        if _F1 > 0.96:
            state = {
            'net': net.state_dict(),
            'F1': _F1,
            'epoch': epoch,
            }
            # best_eval_F1 = _F1
            if not os.path.isdir('/aidata/Ly61/number5/CL0322/ckpt/{}/{}'.format(args.Dataset,args.version)):
                os.makedirs('/aidata/Ly61/number5/CL0322/ckpt/{}/{}'.format(args.Dataset,args.version))
            torch.save(state, '/aidata/Ly61/number5/CL0322/ckpt/{}/{}/ckpt_{}.pth'.format(args.Dataset,args.version,epoch))

        
        return {
                'eval/top-1-acc': f"{top1:.4f}",
                'eval/precision': f"{precision:.4f}",
                'eval/recall': f"{recall:.4f}",
                'eval/F1': f"{_F1:.4f}",
                'AUC/F1': f"{AUC:.4f}"
            }



if __name__ == "__main__":
    
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    random.seed(1)
    parser = argparse.ArgumentParser(description='CL')
    parser.add_argument('--batch-size', type=int, default=100, help='Input batch size for training (default: 32)')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate 0.01') 
    parser.add_argument('--resume', '-r', type=str, default=False,
                        help='resume from checkpoint')
    parser.add_argument('--Dataset', type=str, default='FAST', help=' dataset name (default: FAST or HTRU)')
    parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data (default: /data)')
    parser.add_argument("--train_pulsar", type=int, default=600, help="")
    parser.add_argument("--train_unpulsar", type=int, default=1000, help="")
    parser.add_argument("--val_pulsar", type=int, default=450, help="")
    parser.add_argument("--val_unpulsar", type=int, default=500, help="")
    parser.add_argument("--test_pulsar", type=int, default=1, help="")
    parser.add_argument("--test_unpulsar", type=int, default=1, help="")
    
    # model
    # parser.add_argument("--helps", type=str, default='profile', help="")
    parser.add_argument("--version", type=str, default='080501', help="")
    parser.add_argument("--in_channels", type=int, default=1, help="")
    parser.add_argument("--final_out", type=int, default=1, help="")
    
    # profile
    parser.add_argument("--profile_in", type=int, default=1, help="")
    parser.add_argument("--profile_hidden", type=int, default=512, help="")
    parser.add_argument("--profile_out", type=int, default=2, help="")
    parser.add_argument("--profile_linear", type=int, default=600, help="FAST:1240")
    parser.add_argument("--profile_lengh", type=int, default=64, help="")
    
    # DM
    parser.add_argument("--DM_in", type=int, default=1, help="")
    parser.add_argument("--DM_hidden", type=int, default=512, help="")
    parser.add_argument("--DM_out", type=int, default=2, help="")
    parser.add_argument("--DM_linear", type=int, default=1240, help="FAST:3810")
    parser.add_argument("--DM_lengh", type=int, default=128, help="")
    
    # subband
    parser.add_argument("--subband_out", type=int, default=2, help="")
    
    # subint
    parser.add_argument("--subint_out", type=int, default=2, help="")
    
    # VAE
    parser.add_argument("--latent_dim", type=int, default=64, help="")
    
    # ELBO
    parser.add_argument("--beta", type=int, default=1, help="")  # 需要调参
    

    
    
    args = parser.parse_args()
    
    # 配置日志记录器
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='/aidata/Ly61/number5/CL0322/{}_{}.log'.format(args.Dataset,args.version), filemode='a')

    # 创建一个logger
    logger = logging.getLogger(__name__)
    
    # 定义一个长字符串作为分隔符
    separator = '*' * 100
    big_separator = f"\n{separator}\n{separator}\n{separator}\n{separator}\n{separator}\n"
    # 在你需要突出分隔的地方插入这个分隔字符串
    logger.info(big_separator + "此处是新的日志部分的开始" + big_separator)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 1  # start from epoch 0 or last checkpoint epoch
    
    logger.info(args)

    # New Data
    logger.info('==> Preparing data..')
    print('==> Preparing data..')
    args.dataset_path = '/aidata/Ly61/{}-images-split'.format(args.Dataset)
    # args.dataset_path = '/aidata/Ly61/{}-images'.format(args.Dataset)

    pulsar_dataset = MultiFeatureDataset(args.dataset_path)

    args.loader = PulsarDataLoaderManager(args,pulsar_dataset)
    # subset = args.loader.create_subsets(pulsar_dataset)
    trainloader, testloader, valloader = args.loader.get_dataloaders()
    
    # Model
    logger.info('==> Building model..')
    print('==> Building model..')
    net = MultiModalModel(args=args)
    net.to(device)
    


    # Loss function
    # weights = torch.tensor([100.0, 1.0]).to(device)  # 假设第一类是少数类，给予它更高的权重
    # criterion = nn.CrossEntropyLoss(weight=weights)
    
    criterion = FocalLoss(alpha=0.25, gamma=4) # 2
    # criterion = EqualizedFocalLoss(alpha=0.25, gamma=2, beta=0.99)
    train_size = len(trainloader.dataset)
    elbo = ELBO(net, train_size, args.beta)

    epochs = []
    test_new_accs = []
    test_old_accs = []
    train_losses = []




    ## train step
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100)
    # lambda_lr = lambda epoch: 0.1 if epoch >= 60 else 1  # 在epoch>=10时，学习率乘以10

    # 使用LambdaLR
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    
    best_eval_F1 = 0
    print('==> begin training...')
    for epoch in range(start_epoch, start_epoch+200):
        train_loss = train(epoch)
        # scheduler.step()
        logger.info('==> test model..')
        result = test(epoch)
        logger.info(result)
        epochs.append(epoch)


