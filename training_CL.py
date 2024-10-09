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
from multimodelCombineVAE import MultiModalModel

import matplotlib.pyplot as plt
import pdb
# import transforms
# from dataset import CUB_200_2011_Train, CUB_200_2011_Test
import torchvision.transforms as tfs
import torchvision.datasets as datasets
from sklearn.metrics import *
import logging



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

def train_nonVAE(epoch):
    logger.info('\nEpoch: %d' % epoch)
    net.train()
    net2.eval()

    for batch_idx, (inputs, targets) in enumerate(trainloader2):

                    
        # inputs1, targets1 = inputs1.to(device), targets1.to(device)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)  # [pred,output1,recons,output2,out]        

        # 监督损失
        sup_loss1 = criterion(outputs[-1], targets)

        # outputs_S = F.softmax(outputs[-1]/T,dim=1)
        sup_loss2 = criterion(outputs[0],targets) # beginmodel
        
        sup_loss_vae = criterion(outputs[-2],targets) #
  
        loss_VAE = net.model4.incremental_vae_loss(outputs[2],net2,alpha=args.w5,M_N=0.01)['loss'] # recons VAE
        
        
        loss = args.w1 * sup_loss1 + args.w2 * sup_loss2 + args.w3 * sup_loss_vae + args.w4 * loss_VAE

        loss.backward()
        optimizer.step()
        net.zero_grad()
        

        # _, predicted = outputs[-1].max(1)
        

        if batch_idx % (len(trainloader)*5) == 0:
            # logger.info(f'sup_loss1: {sup_loss1}  loss_VAE: {loss_VAE}  sup_loss2: {sup_loss2}')
            logger.info(f'loss: {loss} sup_loss1: {sup_loss1} sup_loss2: {sup_loss2} sup_loss_vae: {sup_loss_vae} \n loss_VAE: {loss_VAE}')
    # scheduler.step()

    return loss

def train_replay(epoch):
    logger.info('\nEpoch: %d' % epoch)
    net.train()
    net2.eval()

    labeled_iter = iter(trainloader2)
    for batch_idx, (labeled_inputs, labeled_targets) in enumerate(trainloader):
        
        try:
            # 获取有标签数据的下一批次
            inputs, targets = next(labeled_iter)
        except StopIteration:
            # 如果有标签数据迭代完毕，重新启动迭代器
            labeled_iter = iter(trainloader2)
            inputs, targets = next(labeled_iter)
                    
        # inputs1, targets1 = inputs1.to(device), targets1.to(device)
        inputs, targets = inputs.to(device), targets.to(device)
        labeled_inputs, labeled_targets = labeled_inputs.to(device),labeled_targets.to(device)
        
        input = torch.cat((inputs,labeled_inputs))
        target = torch.cat((targets,labeled_targets))

        

        optimizer.zero_grad()
        outputs = net(input)  # [pred,output1,recons,output2,out]

        # 监督损失
        sup_loss1 = criterion(outputs[-1], target)

        # outputs_S = F.softmax(outputs[-1]/T,dim=1)
        sup_loss2 = criterion(outputs[0],target) # beginmodel
        
        sup_loss_vae = criterion(outputs[-2],target) #
  
        loss_VAE = net.model4.incremental_vae_loss(outputs[2],net2,alpha=args.w5,M_N=0.01)['loss'] # recons VAE
        
        
        loss = args.w1 * sup_loss1 + args.w2 * sup_loss2 + args.w3 * sup_loss_vae + args.w4 * loss_VAE

        loss.backward()
        optimizer.step()
        net.zero_grad()
        

        # _, predicted = outputs[-1].max(1)
        

        if batch_idx % (len(trainloader)*5) == 0:
            # logger.info(f'sup_loss1: {sup_loss1}  loss_VAE: {loss_VAE}  sup_loss2: {sup_loss2}')
            logger.info(f'loss: {loss} sup_loss1: {sup_loss1}  sup_loss2: {sup_loss2} sup_loss_vae: {sup_loss_vae} \n loss_VAE: {loss_VAE}')
    # scheduler.step()

    return loss


def train(epoch):
    logger.info('\nEpoch: %d' % epoch)
    net.train()
    net2.eval()

    labeled_iter = iter(trainloader2)
    for batch_idx, (unlabeled_inputs, _) in enumerate(train_unlabeled_loader2):
        
        try:
            # 获取有标签数据的下一批次
            inputs, targets = next(labeled_iter)
        except StopIteration:
            # 如果有标签数据迭代完毕，重新启动迭代器
            labeled_iter = iter(trainloader2)
            inputs, targets = next(labeled_iter)
                    
        # inputs1, targets1 = inputs1.to(device), targets1.to(device)
        inputs, targets = inputs.to(device), targets.to(device)
        unlabeled_inputs = unlabeled_inputs.to(device)
        
        num_lb = inputs.shape[0]
        input = torch.cat((inputs,unlabeled_inputs))

        

        optimizer.zero_grad()
        outputs = net(input)  # [pred,output1,recons,output2,out]
        outputs2 = net2(inputs)
        
        logits_outputs = [sublist[:num_lb] for sublist in outputs]
        logits_unlabeled_outputs = [sublist[num_lb:] for sublist in outputs]
        
        net2_logits_outputs = [sublist2[:num_lb] for sublist2 in outputs2]

        # 监督损失
        sup_loss1 = criterion(logits_outputs[-1], targets)
        
        net2_sup_loss1 = criterion(net2_logits_outputs[-1], targets)
        # outputs_S = F.softmax(outputs[-1]/T,dim=1)
        sup_loss2 = criterion(logits_outputs[0],targets) # beginmodel
        
        sup_loss_vae = criterion(logits_outputs[-2],targets) #

        
        # 伪标签数据损失
        pseudo_labels1,pseudo_labels2 = logits_unlabeled_outputs[-1].argmax(dim=1),logits_unlabeled_outputs[0].argmax(dim=1)
        probs1,probs2 = torch.softmax(logits_unlabeled_outputs[-1], dim=1),torch.softmax(logits_unlabeled_outputs[0], dim=1)
        weights1,weights2 = compute_weights(probs1, args.mu_t, args.sigma_t, args.lambda_max),compute_weights(probs2, args.mu_t, args.sigma_t, args.lambda_max)
        unsup_loss1 = (criterion(logits_unlabeled_outputs[-1], pseudo_labels1) * weights1).mean()
        # unsup_loss2 = (criterion(logits_unlabeled_outputs[0], pseudo_labels2) * weights2).mean()
  
        loss_VAE = net.model4.incremental_vae_loss(outputs[2],net2,alpha=args.w5,M_N=0.01)['loss'] # recons VAE
        
        
        loss = args.w1 * sup_loss1 + args.w2 * sup_loss2 + args.w3 * sup_loss_vae + args.w4 * loss_VAE + args.w6 * unsup_loss1

        loss.backward()
        optimizer.step()
        net.zero_grad()
        

        # _, predicted = outputs[-1].max(1)
        

        if batch_idx % (len(trainloader)*5) == 0:
            # logger.info(f'sup_loss1: {sup_loss1}  loss_VAE: {loss_VAE}  sup_loss2: {sup_loss2}')
            logger.info(f'loss: {loss} sup_loss1: {sup_loss1} net2_sup_loss1: {net2_sup_loss1} sup_loss2: {sup_loss2} sup_loss_vae: {sup_loss_vae} \n loss_VAE: {loss_VAE}  unsup_loss1: {unsup_loss1}')
    # scheduler.step()

    return loss

def test(epoch):
    best_eval_F1 = 0
    net.eval()
    y_true = []
    y_pred = []
    y_logits = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader2):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs) #[pred,output1,recons,output2,out]

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
        
        state = {
        'net': net.state_dict(),
        'F1': _F1,
        'epoch': epoch,
        }
        best_eval_F1 = _F1
        if not os.path.isdir('/aidata/Ly61/number5/CL0322/ckpt/{}/{}'.format(args.Dataset,args.version)):
            os.makedirs('/aidata/Ly61/number5/CL0322/ckpt/{}/{}'.format(args.Dataset, args.version), exist_ok=True)
        torch.save(state, '/aidata/Ly61/number5/CL0322/ckpt/{}/{}/ckpt_HTRU_CL_{}.pth'.format(args.Dataset,args.version,epoch))

        
        return {'eval/top-1-acc': top1, 'eval/precision': precision, 'eval/recall': recall, 'eval/F1': _F1, 'AUC/F1': AUC}
    
def test_pre():
    best_eval_F1 = 0
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
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs) #[pred,output1,recons,output2,out]

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
        
            logger.info(f'epoch: {epoch} eval/top-1-acc: {top1}, eval/precision: {precision}, eval/recall: {recall}, eval/F1: {_F1}, AUC/F1: {AUC}')

def get_parameters(net, vae_lr, base_lr):
    base_params = filter(lambda p: p.requires_grad, net.parameters())
    vae_params = filter(lambda p: p.requires_grad, net.model4.parameters())
    
    
    params = [
        {'params': base_params, 'lr': base_lr},
        {'params': vae_params, 'lr': vae_lr},
    ]
    
    return params

def copy_layer_params(old_model, new_model, layer_name):
    """
    将旧模型的层参数覆盖到新模型的相应层。
    
    old_model: 旧模型实例
    new_model: 新模型实例
    layer_name: 要复制参数的层的名称
    """
    old_layer = dict(old_model.named_modules())[layer_name]
    new_layer = dict(new_model.named_modules())[layer_name]
    
    # 检查两个层之间的结构是否相同
    for old_param, new_param in zip(old_layer.parameters(), new_layer.parameters()):
        if old_param.data.shape == new_param.data.shape:
            new_param.data = old_param.data.clone()  # 复制参数
        else:
            raise ValueError('Layer structure is not the same, cannot copy parameters')

if __name__ == "__main__":
    
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    random.seed(1)
    parser = argparse.ArgumentParser(description='CL')
    parser.add_argument('--batch-size', type=int, default=1024, help='Input batch size for training (default: 32)')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', type=str, default=False,
                        help='resume from checkpoint')
    parser.add_argument('--Dataset', type=str, default='HTRU', help=' dataset name (default: FAST or HTRU)')
    parser.add_argument('--version', type=str, default='040902', help=' Version')

    parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data (default: /data)')
    parser.add_argument("--train_pulsar", type=int, default=600, help="")
    parser.add_argument("--train_unpulsar", type=int, default=600, help="")
    parser.add_argument("--val_pulsar", type=int, default=500, help="")
    parser.add_argument("--val_unpulsar", type=int, default=500, help="")
    parser.add_argument("--test_pulsar", type=int, default=1, help="")
    parser.add_argument("--test_unpulsar", type=int, default=1, help="")
    
    # model
    parser.add_argument("--in_channels", type=int, default=1, help="")
    
    # Loss weight
    parser.add_argument("--w1", type=float, default=0.10, help="begin model")
    parser.add_argument("--w2", type=float, default=0.10, help="vae model")
    parser.add_argument("--w3", type=float, default=0.50, help="all model")
    parser.add_argument("--w4", type=float, default=0.10, help="recon and kl")
    parser.add_argument("--w5", type=float, default=1.0, help="consistency")
    parser.add_argument("--w6", type=float, default=0.15, help="unlabel all model")
    parser.add_argument("--w7", type=float, default=0.5, help="old model")
    
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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='/aidata/Ly61/number5/CL0322/HTRU_CL_{}.log'.format(args.version), filemode='a')

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
    
    ######## dataset1 ##############
    args.dataset_path = '/aidata/Ly61/FAST-images'
    pulsar_dataset = PulsarDataset(args.dataset_path,use_vae_data=args.use_vae_data,vae_number=args.vae_number)

    args.loader = PulsarDataLoaderManager(args,pulsar_dataset)
    subset = args.loader.create_subsets(pulsar_dataset)
    trainloader, testloader, valloader = args.loader.get_dataloaders(subset)
    
    # _,unlabel_subset = args.loader.get_unlabeled(pulsar_dataset,num_unlabeled_samples=args.vae_number)  # 无标签非脉冲星数量：num_unlabeled_samples
    # train_unlabeled_loader = args.loader.get_train_unlabeled_loader(unlabel_subset,batch_size=args.batch_size)

    
    ######## dataset2 ##############
    args.dataset_path2 = '/aidata/Ly61/HTRU-images'
    pulsar_dataset2 = PulsarDataset(args.dataset_path2,use_vae_data=args.use_vae_data,vae_number=args.vae_number)

    args.loader2 = PulsarDataLoaderManager(args,pulsar_dataset2)
    subset2 = args.loader2.create_subsets(pulsar_dataset2)
    trainloader2, testloader2,valloader2 = args.loader2.get_dataloaders(subset2)
    
    _,unlabel_subset2 = args.loader2.get_unlabeled(pulsar_dataset2,num_unlabeled_samples=args.vae_number)
    train_unlabeled_loader2 = args.loader2.get_train_unlabeled_loader(unlabel_subset2,batch_size=args.batch_size)

    logger.info(args)
    # Model
    logger.info('==> Building model..')
    print('==> Building model..')
    net = MultiModalModel(num_classes=2,args=args)
    net.to(device)
    
    net2 = MultiModalModel(num_classes=2,args=args) # old模型
    net2.to(device)
    
    oor = torch.load('/aidata/Ly61/number5/CL0322/ckpt/ckpt_FAST_0404.pth')

    # 'Net'  is the new model to learn new classes
    net.load_state_dict(oor['net'])
    net2.load_state_dict(oor['net'])
    
    
    # # 冻结所有层
    
    # for param in net.modal1.parameters():
    #     param.requires_grad = False
        
    for param in net.parameters():
        param.requires_grad = False
    
    # for param in net.model4.parameters():
    #     param.requires_grad = True


    # layers_to_unfreeze = ['conv4_x', 'conv5_x']
    # layers_to_unfreeze = ['conv1_x','conv2_x']
    layers_to_unfreeze = ['fc']
    
    # # # 分别对每个模块应用unfreeze_selected_layers函数
    unfreeze_selected_layers(net.modal1, layers_to_unfreeze)
    unfreeze_selected_layers(net.modal2, layers_to_unfreeze)
    unfreeze_selected_layers(net.modal3, layers_to_unfreeze)
    # unfreeze_selected_layers(net.model4, ['decoder_input'])
    
    # 冻结所有层
    for param in net2.parameters():
        param.requires_grad = False


    # Loss function
    criterion = nn.CrossEntropyLoss()

    epochs = []
    test_new_accs = []
    test_old_accs = []
    train_losses = []


    # 使用帮助函数获取参数
    # params = get_parameters(net, args.lr/10, args.lr)

    ## train step
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4) # 5e-4

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), weight_decay=5e-4)

    # optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=5e-4, momentum=0.9)

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=0)
    print('==> begin training...')
    for epoch in range(start_epoch, start_epoch+100):
        train_loss = train(epoch)
        result = test(epoch)
        logger.info(result)
        epochs.append(epoch)
    logger.info('==> Final FAST Test')
    print('==> Final FAST Test')
    test_pre()



