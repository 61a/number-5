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
# from utils import progress_bar
import numpy as np
import random
from data import PulsarDataset,PulsarDataLoaderManager
# from mutimodel import MultiModalModel
from miltimodelCode import MultiModalModel

import matplotlib.pyplot as plt
import pdb
# import transforms
# from dataset import CUB_200_2011_Train, CUB_200_2011_Test
import torchvision.transforms as tfs
import torchvision.datasets as datasets
from sklearn.metrics import *
import logging




def kaiming_normal_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, nn.Linear):
		nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        # predictions = torch.argmax(log_pred, dim=1)
        # print(targets[:20],log_pred[:20])
        # recall = recall_score(targets.cpu().numpy(), predictions.cpu().detach().numpy(), average='binary')
        # return nn.CrossEntropyLoss(log_pred, targets, reduction=reduction)
        # return F.nll_loss(log_pred, targets, reduction=reduction)
        return F.cross_entropy(logits, targets, reduction=reduction) #this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss

# 计算权重
def compute_weights(probs, mu_t=0.90, sigma_t=0.1, lambda_max=0.80):
    max_probs = probs.max(dim=1)[0]
    weights = torch.where(
        max_probs < mu_t,
        lambda_max * torch.exp(-((max_probs - mu_t)**2) / (2 * sigma_t**2)),
        torch.tensor([lambda_max]).to(probs.device)
    )
    # min_max_probs = max_probs.min()
    # max_max_probs = max_probs.max()
    # with open('max_probs_stats.txt', 'a') as f:
    #     f.write(f'max_probs Min: {min_max_probs}  ')
    #     f.write(f'max_probs Max: {max_max_probs}\n')
    return weights


def train(epoch):
    logger.info('Epoch: %d' % epoch)
    net.train()
    # 将有标签数据的DataLoader转换为迭代器
    labeled_iter = iter(trainloader)

    for batch_idx, (unlabeled_inputs, _) in enumerate(train_unlabeled_loader):
        
        try:
            # 获取有标签数据的下一批次
            inputs, targets = next(labeled_iter)
        except StopIteration:
            # 如果有标签数据迭代完毕，重新启动迭代器
            labeled_iter = iter(trainloader)
            inputs, targets = next(labeled_iter)
            
        inputs, targets = inputs.to(device), targets.to(device)
        unlabeled_inputs = unlabeled_inputs.to(device)
        
        num_lb = inputs.shape[0]
        input = torch.cat((inputs,unlabeled_inputs))

        optimizer.zero_grad()
        outputs = net(input)  # [o1,o2,o3,o]
        logits_outputs = [sublist[:num_lb] for sublist in outputs]
        logits_unlabeled_outputs = [sublist[num_lb:] for sublist in outputs]

        # 监督损失
        sup_loss1 = ce_loss(logits_outputs[-1], targets, reduction='mean')
        # sup_loss2 = criterion(logits_outputs[0],targets) # beginmodel
        
        # 伪标签数据损失
        pseudo_labels1 = logits_unlabeled_outputs[-1].argmax(dim=1)
        probs1 = torch.softmax(logits_unlabeled_outputs[-1], dim=1)
        weights1 = compute_weights(probs1, args.mu_t, args.sigma_t, args.lambda_max)
        unsup_loss1 = (ce_loss(logits_unlabeled_outputs[-1], pseudo_labels1) * weights1).mean()
   

        loss_VAE1 = net.modal1.loss_function(outputs[0],M_N=0.001)['loss'] # recons VAE
        loss_VAE2 = net.modal2.loss_function(outputs[1],M_N=0.001)['loss']
        loss_VAE3 = net.modal3.loss_function(outputs[2],M_N=0.001)['loss']
        
        
        loss = sup_loss1 + 0.05*(loss_VAE1 + loss_VAE2 + loss_VAE3) + unsup_loss1 

        loss.backward()
        optimizer.step()
        net.zero_grad()

        # _, predicted = outputs[-1].max(1)
        

        if batch_idx % len(trainloader)*1000 == 0:
            logger.info(f'sup_loss1: {sup_loss1}  unsup_loss1: {unsup_loss1}  loss_VAE1: {loss_VAE1}  loss_VAE2: {loss_VAE2}  loss_VAE3: {loss_VAE3}')


    return loss

def test(epoch):
    best_eval_F1 = 0
    net.eval()
    y_true = []
    y_pred = []
    y_logits = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs) # [o1,o2,o3,o]

            y_true.extend(targets.cpu().tolist())
            y_pred.extend(torch.max(outputs[-1], dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(outputs[-1], dim=-1).cpu().tolist())
        
        top1 = accuracy_score(y_true, y_pred)
        #top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        #top5 = 'HTRU none'
        precision = precision_score(y_true, y_pred, average='binary',pos_label=0)
        recall = recall_score(y_true, y_pred, average='binary',pos_label=0)
        _F1 = f1_score(y_true, y_pred, average='binary',pos_label=0)
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
            if not os.path.isdir('/aidata/Ly61/number5/CL0322/ckpt'):
                os.mkdir('/aidata/Ly61/number5/CL0322/ckpt')
            torch.save(state, '/aidata/Ly61/number5/CL0322/ckpt/ckpt_{}_{}_{}.pth'.format(args.Dataset,'code',args.in_channels))

        
        return {'eval/top-1-acc': top1, 'eval/precision': precision, 'eval/recall': recall, 'eval/F1': _F1, 'AUC/F1': AUC}


if __name__ == "__main__":
    
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    random.seed(1)
    parser = argparse.ArgumentParser(description='CL')
    parser.add_argument('--batch-size', type=int, default=1024, help='Input batch size for training (default: 32)')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', type=str, default=False,
                        help='resume from checkpoint')
    parser.add_argument('--Dataset', type=str, default='FAST', help=' dataset name (default: FAST or HTRU)')
    parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data (default: /data)')
    parser.add_argument("--train_pulsar", type=int, default=600, help="")
    parser.add_argument("--train_unpulsar", type=int, default=600, help="")
    parser.add_argument("--val_pulsar", type=int, default=500, help="")
    parser.add_argument("--val_unpulsar", type=int, default=500, help="")
    parser.add_argument("--test_pulsar", type=int, default=1, help="")
    parser.add_argument("--test_unpulsar", type=int, default=1, help="")
    
    # model
    parser.add_argument("--in_channels", type=int, default=1, help="")
    
    # unlabel weight
    parser.add_argument("--lambda_max", type=float, default=0.80, help="")
    parser.add_argument("--mu_t", type=float, default=0.90, help="")
    parser.add_argument("--sigma_t", type=float, default=0.10, help="")
    
    # VAE
    parser.add_argument("--latent_dim", type=int, default=64, help="")
    parser.add_argument("--vae_number", type=int, default=20000, help="")
    parser.add_argument('--use_vae_data', type=str, default=False)
    
    
    args = parser.parse_args()
    
    # 配置日志记录器
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='/aidata/Ly61/number5/CL0322/{}_{}_{}.log'.format(args.Dataset,'code',args.in_channels), filemode='a')

    # 创建一个logger
    logger = logging.getLogger(__name__)
    
    # 定义一个长字符串作为分隔符
    separator = '*' * 200
    big_separator = f"\n{separator}\n{separator}\n{separator}\n{separator}\n{separator}\n{separator}\n{separator}\n"
    # 在你需要突出分隔的地方插入这个分隔字符串
    logger.info(big_separator + "此处是新的日志部分的开始" + big_separator)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 1  # start from epoch 0 or last checkpoint epoch

    # New Data
    logger.info('==> Preparing data..')
    print('==> Preparing data..')
    args.dataset_path = '/aidata/Ly61/number3/{}-profile-submerge'.format(args.Dataset)
    pulsar_dataset = PulsarDataset(args.dataset_path,use_vae_data=args.use_vae_data,vae_number=args.vae_number)

    args.loader = PulsarDataLoaderManager(args,pulsar_dataset)
    subset = args.loader.create_subsets(pulsar_dataset)
    trainloader, testloader, valloader = args.loader.get_dataloaders(subset)
    
    _,unlabel_subset = args.loader.get_unlabeled(pulsar_dataset,num_unlabeled_samples=args.vae_number)  # 无标签非脉冲星数量：num_unlabeled_samples
    train_unlabeled_loader = args.loader.get_train_unlabeled_loader(unlabel_subset,batch_size=args.batch_size)

    # Model
    logger.info('==> Building model..')
    print('==> Building model..')
    net = MultiModalModel(num_classes=2,args=args)
    net.to(device)


    # Loss function
    criterion = nn.CrossEntropyLoss()

    epochs = []
    test_new_accs = []
    test_old_accs = []
    train_losses = []




    ## train step
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=5e-4)

    print('==> begin training...')
    for epoch in range(start_epoch, start_epoch+300):
        train_loss = train(epoch)
        result = test(epoch)
        logger.info(result)
        epochs.append(epoch)


