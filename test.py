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
from multimodel_fourBayesian import MultiModalModel
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

class ELBO(nn.Module):

    def __init__(self, model, train_size, beta):
        super().__init__()
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.beta = beta
        self.train_size = train_size

    def forward(self, outputs, targets, kl):
        assert not targets.requires_grad
        return criterion(outputs.squeeze(),targets)  + self.beta * kl / self.num_params

def distillation_loss(student_logits, teacher_logits, T=2.0):
    """
    计算蒸馏损失
    T: 温度参数，使输出软化
    """
    soft_target = F.softmax(teacher_logits / T, dim=1)
    soft_output = F.log_softmax(student_logits / T, dim=1)
    return F.kl_div(soft_output, soft_target, reduction='batchmean') * (T**2)

# 定义一个辅助函数来选择性地解冻模型中的特定层
def unfreeze_selected_layers(model, layers_to_unfreeze):
    """
    model: 要修改的模型
    layers_to_unfreeze: 一个包含要解冻层名的列表
    """
    for name, child in model.named_children():
        if name in layers_to_unfreeze:
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False


def test():
    directory = '/aidata/Ly61/number5/CL0322/ckpt/{}/{}'.format(args.Dataset,args.version)

    # 列出所有的 .pth 文件
    for filename in os.listdir(directory):
        if 'HTRU' in filename and filename.endswith('.pth'):
            oor2 = torch.load(os.path.join(directory, filename))
            epoch = oor2['epoch']
            if epoch == 12 or epoch ==27:
                net.load_state_dict(oor2['net'])
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

                        outputs = net(profiles, dm_curves, subbands, subints,'task1') #[pred,output1,recons,output2,out]

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

                # 保存 y_true 和 y_logits 到 CSV
                import pandas as pd
                results_df = pd.DataFrame({
                    'y_true': y_true,
                    'y_logits': y_logits
                })
                results_filename = f'/aidata/Ly61/number5/CL0322/ececsv/test/results_test_{filename}.csv'
                results_df.to_csv(results_filename, index=False)
                logger.info(f'Results saved to {results_filename}')
def test_pre():
    directory = '/aidata/Ly61/number5/CL0322/ckpt/{}/{}'.format(args.Dataset,args.version)

    # 列出所有的 .pth 文件
    for filename in os.listdir(directory):
        if 'HTRU' in filename and filename.endswith('.pth'):
            oor2 = torch.load(os.path.join(directory, filename))
            epoch = oor2['epoch']
            if epoch == 12 or epoch ==27:
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

                        outputs = net(profiles, dm_curves, subbands, subints,'task1') #[pred,output1,recons,output2,out]

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
                
                # 保存 y_true 和 y_logits 到 CSV
                import pandas as pd
                results_df = pd.DataFrame({
                    'y_true': y_true,
                    'y_logits': y_logits
                })
                results_filename = f'/aidata/Ly61/number5/CL0322/ececsv/testpre/results_testpre_{filename}.csv'
                results_df.to_csv(results_filename, index=False)
                logger.info(f'Results saved to {results_filename}')

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
    parser.add_argument('--version', type=str, default='050213', help=' Version')
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

    # ELBO
    parser.add_argument("--beta", type=int, default=0.01, help="")  # 需要调参
    
    
    args = parser.parse_args()
    
    # 配置日志记录器
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='/aidata/Ly61/number5/CL0322/{}_CL_Test.log'.format(args.Dataset), filemode='a')

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
    
    

    # 设置随机种子
    random.seed(42)  # 使用42作为种子值，你可以选择任意值

    # 生成一百个随机整数
    random_numbers = [random.randint(1, 100) for _ in range(100)]

    # 打印随机数
    for number in random_numbers:
        logger.info(big_separator)
        logger.info(f'randon_number==> {number}')
        ######## dataset1 ##############
        args.dataset_path = '/aidata/Ly61/FAST-images-split'
        pulsar_dataset = MultiFeatureDataset(args.dataset_path)

        args.loader = PulsarDataLoaderManager(args,pulsar_dataset,random_seed=number)
        
        trainloader, testloader, valloader = args.loader.get_dataloaders()
        
        ######## dataset2 ##############
        args.dataset_path2 = '/aidata/Ly61/HTRU-images-split'
        pulsar_dataset2 = MultiFeatureDataset(args.dataset_path2)

        args.loader2 = PulsarDataLoaderManager(args,pulsar_dataset2,random_seed=number)
        
        trainloader2, testloader2,valloader2 = args.loader2.get_dataloaders()
        
        #########################################################################################################
        #########################################################################################################
        #########################################################################################################
        
        logger.info(args)
        # Model
        logger.info('==> Building model..')
        print('==> Building model..')
        
        Tem = 2
        logger.info(f'T==> {Tem}')
        net = MultiModalModel(args=args)
        net.to(device)
        
        # oor = torch.load('/aidata/Ly61/number5/CL0322/ckpt/FAST/050211/ckpt_best.pth')

        # # 'Net'  is the new model to learn new classes
        # net.load_state_dict(oor['net'])
        logger.info('#######################################################################################################')
        logger.info('==> Final HTRU Test')
        print('==> Final HTRU Test')
        test()
        logger.info('#######################################################################################################')

        logger.info('==> Final FAST Test')
        print('==> Final FAST Test')
        test_pre()



