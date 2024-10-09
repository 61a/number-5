import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, ConcatDataset
import scipy.io as scio
from scipy.io import loadmat
import torchvision
import os
import argparse
# from utils import progress_bar
import numpy as np
import random
from data_four import MultiFeatureDataset, PulsarDataLoaderManager
from multimodel_four import MultiModalModel
from loss import FocalLoss, EqualizedFocalLoss
from util.ewc import EWC

import matplotlib.pyplot as plt
import pdb
# import transforms
# from dataset import CUB_200_2011_Train, CUB_200_2011_Test
import torchvision.transforms as tfs
import torchvision.datasets as datasets
from sklearn.metrics import *
import logging



# 定义一个辅助函数来选择性地冻模型中的特定层
def unfreeze_selected_layers(model, layers_to_unfreeze):
    """
    model: 要修改的模型
    layers_to_unfreeze: 一个包含要冻层名的列表
    """
    for name, child in model.named_children():
        # print(name)
        if name in layers_to_unfreeze:
            for param in child.parameters():
                param.requires_grad = False
        else:
            for param in child.parameters():
                param.requires_grad = True

def train(epoch):
    logger.info('Epoch: %d' % epoch)
    net.train()
    net2.eval()

    for batch_idx, data  in enumerate(trainloader2):
        
        profiles = data['cand_profile'].to(device)
        dm_curves = data['cand_dm_curve'].to(device)
        subbands = data['cand_subbands'].to(device)
        subints = data['cand_subints'].to(device)
        targets = data['label'].to(device).float()       

        
        with torch.no_grad():
            outputs_old = net2(profiles, dm_curves, subbands, subints,'task1').detach()
            
        optimizer.zero_grad()
        outputs = net(profiles, dm_curves, subbands, subints,'task2')  
        
        loss_new = criterion(outputs.squeeze(),targets) 
        # loss_old = criterion(outputs_old.squeeze(),targets) 
        loss_distill = criterion_distill(nn.functional.log_softmax(outputs, dim=1), nn.functional.softmax(outputs_old, dim=1))
        # ewc_loss = ewc_regularizer.regularize(net.named_parameters())
        # loss = loss_old + loss_new + loss_distill
        # loss = loss_new + loss_distill + ewc_loss
        # loss = loss_new + kl_loss
        loss = loss_new + loss_distill
        
        loss.backward()
        optimizer.step()
        net.zero_grad()
        

        if batch_idx % (len(trainloader)*5) == 0:
            # logger.info(f'loss: {loss:.6f} loss_old: {loss_old:.6f} loss_new: {loss_new:.6f} loss_distill: {loss_distill:.6f} ewc_loss: {ewc_loss:.6f}')
            # logger.info(f'loss: {loss.item():.6f} loss_new: {loss_new.item():.6f} loss_old: {loss_old:.6f} loss_distill: {loss_distill:.6f}')
            logger.info(f'loss: {loss.item():.6f} loss_new: {loss_new.item():.6f} loss_distill: {loss_distill:.6f}')

    # scheduler.step()

    return loss

def test(epoch):
    global best_eval_F1
    net.eval()
    y_true = []
    y_pred = []
    y_logits = []
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader2):
            profiles = data['cand_profile'].to(device)
            dm_curves = data['cand_dm_curve'].to(device)
            subbands = data['cand_subbands'].to(device)
            subints = data['cand_subints'].to(device)
            targets = data['label'].to(device)

            outputs = net(profiles, dm_curves, subbands, subints,'task2') #[pred,output1,recons,output2,out]

            # 这里使用sigmoid函数因为是二分类问题
            probs = torch.sigmoid(outputs.squeeze()).cpu()
            preds = (probs > 0.5).int()  # 使用0.5作为阈值
            
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(preds.tolist())
            y_logits.extend(probs.tolist())
        
        top1 = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary',pos_label=1, zero_division=1)
        recall = recall_score(y_true, y_pred, average='binary',pos_label=1)
        _F1 = f1_score(y_true, y_pred, average='binary',pos_label=1)       
        AUC = roc_auc_score(y_true, y_pred, multi_class='ovo')
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
                os.makedirs('/aidata/Ly61/number5/CL0322/ckpt/{}/{}'.format(args.Dataset, args.version), exist_ok=True)
            torch.save(state, '/aidata/Ly61/number5/CL0322/ckpt/{}/{}/ckpt_HTRU_CL_{}.pth'.format(args.Dataset,args.version,epoch))
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
    
def test_pre():
    directory = '/aidata/Ly61/number5/CL0322/ckpt/{}/{}'.format(args.Dataset,args.version)

    # 列出所有的 .pth 文件
    for filename in os.listdir(directory):
        if filename.endswith('.pth'):
            oor2 = torch.load(os.path.join(directory, filename))
            epoch = oor2['epoch']
            net.load_state_dict(oor2['net'])
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

                    outputs = net(profiles, dm_curves, subbands, subints,'task1') 

                    # 这里使用sigmoid函数因为是二分类问题
                    probs = torch.sigmoid(outputs.squeeze()).cpu()
                    preds = (probs > 0.5).int()  # 使用0.5作为阈值
                    
                    y_true.extend(targets.cpu().tolist())
                    y_pred.extend(preds.tolist())
                    y_logits.extend(probs.tolist())
                
                top1 = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='binary',pos_label=1, zero_division=1)
                recall = recall_score(y_true, y_pred, average='binary',pos_label=1)
                _F1 = f1_score(y_true, y_pred, average='binary',pos_label=1)               
                AUC = roc_auc_score(y_true, y_pred, multi_class='ovo')
                cf_mat = confusion_matrix(y_true, y_pred, normalize=None)
                logger.info(f'confusion matrix:\n {np.array_str(cf_mat)}')
        
            logger.info(f'epoch: {epoch} eval/top-1-acc: {top1:.4f}, eval/precision: {precision:.4f}, eval/recall: {recall:.4f}, eval/F1: {_F1:.4f}, AUC/F1: {AUC:.4f}')


def random_initialize_layers(layers):
    for layer in layers:
        if hasattr(layer, 'weight'):
            layer.weight.data = torch.nn.init.normal_(layer.weight.data, mean=0.0, std=0.1)
        if hasattr(layer, 'bias'):
            layer.bias.data.fill_(0.01)

if __name__ == "__main__":
    
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    random.seed(1)
    parser = argparse.ArgumentParser(description='CL')
    parser.add_argument('--batch-size', type=int, default=128, help='Input batch size for training (default: 32)')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', type=str, default=False,
                        help='resume from checkpoint')
    parser.add_argument('--Dataset', type=str, default='HTRU', help=' dataset name (default: FAST or HTRU)')
    

    parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data (default: /data)')
    parser.add_argument("--train_pulsar", type=int, default=600, help="")
    parser.add_argument("--train_unpulsar", type=int, default=1000, help="")
    parser.add_argument("--val_pulsar", type=int, default=500, help="")
    parser.add_argument("--val_unpulsar", type=int, default=500, help="")
    parser.add_argument("--test_pulsar", type=int, default=1, help="")
    parser.add_argument("--test_unpulsar", type=int, default=1, help="")
    
    # model
    # parser.add_argument("--helps", type=str, default='profile', help="")
    parser.add_argument('--version', type=str, default='050401', help=' Version')
    parser.add_argument("--in_channels", type=int, default=1, help="")
    parser.add_argument("--final_out", type=int, default=1, help="")
    
    # EWC
    parser.add_argument("--lamda", type=int, default=100, help="")
    
    # profile
    parser.add_argument("--profile_in", type=int, default=1, help="")
    parser.add_argument("--profile_hidden", type=int, default=512, help="")
    parser.add_argument("--profile_out", type=int, default=2, help="")
    parser.add_argument("--profile_linear", type=int, default=600, help="FAST:600")
    parser.add_argument("--profile_lengh", type=int, default=64, help="")
    
    # DM
    parser.add_argument("--DM_in", type=int, default=1, help="")
    parser.add_argument("--DM_hidden", type=int, default=512, help="")
    parser.add_argument("--DM_out", type=int, default=2, help="")
    parser.add_argument("--DM_linear", type=int, default=1240, help="FAST:1240")
    parser.add_argument("--DM_lengh", type=int, default=128, help="")
    
    # subband
    parser.add_argument("--subband_out", type=int, default=2, help="")
    
    # subint
    parser.add_argument("--subint_out", type=int, default=2, help="")
    
    # VAE
    parser.add_argument("--latent_dim", type=int, default=64, help="")

    
    
    args = parser.parse_args()
    
    # 配置日志记录器
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='/aidata/Ly61/number5/CL0322/HTRU_CL_{}.log'.format(args.version), filemode='a')

    # 创建一个logger
    logger = logging.getLogger(__name__)
    
    # 定义一个长字符串作为分隔符
    separator = '*' * 200
    big_separator = f"\n{separator}\n{separator}\n{separator}\n{separator}\n{separator}\n{separator}\n{separator}\n"
    # 在你需要突出分隔的地方插入这个分隔字符串
    logger.info(big_separator + "此处是新的日志部分的开始" + big_separator)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.device = device
    best_acc = 0  # best test accuracy
    start_epoch = 1  # start from epoch 0 or last checkpoint epoch

    # New Data
    logger.info('==> Preparing data..')
    print('==> Preparing data..')
    
    ######## dataset1 ##############
    args.dataset_path = '/aidata/Ly61/FAST-images-split'
    pulsar_dataset = MultiFeatureDataset(args.dataset_path)

    args.loader = PulsarDataLoaderManager(args,pulsar_dataset)
    
    trainloader, testloader, valloader = args.loader.get_dataloaders()
    
    ######## dataset2 ##############
    args.dataset_path2 = '/aidata/Ly61/HTRU-images-split'
    pulsar_dataset2 = MultiFeatureDataset(args.dataset_path2)

    args.loader2 = PulsarDataLoaderManager(args,pulsar_dataset2)
    
    trainloader2, testloader2,valloader2 = args.loader2.get_dataloaders()
    
    # 首先，我们需要获取这两个DataLoader引用的数据集
    dataset1 = trainloader.dataset
    dataset2 = trainloader2.dataset

    # 使用ConcatDataset来合并这两个数据集
    combined_dataset = ConcatDataset([dataset1, dataset2])

    # 最后根据合并后的数据集创建一个新的DataLoader
    # 你可以根据需要，设置batch_size、shuffle等参数
    combined_trainloader = DataLoader(combined_dataset, batch_size=64, shuffle=True)
    
    #########################################################################################################
    #########################################################################################################
    #########################################################################################################
    
    logger.info(args)
    # Model
    logger.info('==> Building model..')
    print('==> Building model..')
    net = MultiModalModel(args=args)
    net.to(device)
    
    net2 = MultiModalModel(args=args)# old模型
    net2.to(device)
    
    oor = torch.load('/aidata/Ly61/number5/CL0322/ckpt/FAST/050203/ckpt_best.pth')

    # 'Net'  is the new model to learn new classes
    net.load_state_dict(oor['net'])
    net2.load_state_dict(oor['net'])
    # random_initialize_layers([net.model5])
    
    
    
    
    
    # EWC
    
    # ewc_regularizer = EWC(args)
    
    # ewc_regularizer.update_fisher_optpar(net2, trainloader,device)
    # # 冻结所有层
    # for param in net.parameters():
    #     param.requires_grad = False
    # 冻结所有层
    for param in net2.parameters():
        param.requires_grad = False
        
    layers_to_unfreeze1 = ['conv1','norm1','conv2','dropout','fc1']
    layers_to_unfreeze2 = ['pre','stage1','stage2','stage3','stage4']
    # layers_to_unfreeze = ['']
    
    
    # 分别对每个模块应用unfreeze_selected_layers函数
    unfreeze_selected_layers(net.modal1, layers_to_unfreeze1)
    unfreeze_selected_layers(net.modal2, layers_to_unfreeze1)
    unfreeze_selected_layers(net.modal3, layers_to_unfreeze2)
    unfreeze_selected_layers(net.modal4, layers_to_unfreeze2)
    # unfreeze_selected_layers(net.model5_task1, layers_to_unfreeze)
    # unfreeze_selected_layers(net.model5_task2, layers_to_unfreeze)




    # Loss function
    criterion = FocalLoss(alpha=0.25, gamma=4) 
    # 定义蒸馏损失，使用例如Kullback-Leibler散度
    criterion_distill = nn.KLDivLoss(reduction='batchmean')

    epochs = []
    test_new_accs = []
    test_old_accs = []
    train_losses = []

    ## train step
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4) # 5e-4
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), weight_decay=5e-4)   
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=1e-4)

    best_eval_F1 = 0

    print('==> begin training...')
    for epoch in range(start_epoch, start_epoch+100):
        train_loss = train(epoch)
        result = test(epoch)
        logger.info(result)
        epochs.append(epoch)
    logger.info('==> Final FAST Test')
    print('==> Final FAST Test')
    test_pre()



