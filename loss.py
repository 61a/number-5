import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        
        # 计算模型的交叉熵损失
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 根据公式，计算 p_t
        pt = torch.exp(-BCE_loss)
        
        # 计算 Focal Loss
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        # 返回均值作为最终的损失
        return torch.mean(F_loss)
    
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2, ignore_index=-100, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         # 如果alpha为None，将其设为1，即不对类别进行加权
#         self.alpha = alpha if alpha is not None else torch.tensor(1.0)
#         self.ignore_index = ignore_index
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         # 输入应已经包含softmax/log_softmax的计算结果
#         # 首先计算交叉熵损失，不过这里不应用平均(reduction='none'以获取每个样本的损失)
#         CE_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
#         # 获取最大的概率用于后续计算
#         pt = torch.exp(-CE_loss) # 计算pt
#         # 如果alpha是一个列表或张量，则为每个类别应用对应的alpha
#         if isinstance(self.alpha, (list, torch.Tensor)):
#             self.alpha = self.alpha.to(inputs.device)
#             at = self.alpha.gather(0, targets.data.view(-1))
#             CE_loss = at * CE_loss

#         # 计算最终的focal loss
#         F_loss = (1 - pt) ** self.gamma * CE_loss

#         if self.reduction == 'mean':
#             return torch.mean(F_loss)
#         elif self.reduction == 'sum':
#             return torch.sum(F_loss)
#         else:
#             return F_loss

# # 测试 FocalLoss
# criterion = FocalLoss(alpha=0.25, gamma=2)

# # 模拟预测值和目标值
# inputs = torch.randn(10, requires_grad=True)  # 模型预测的原始对数概率（logits）
# targets = torch.empty(10).random_(2)         # 真实目标，随机生成的0或1

# # 计算损失
# loss = criterion(inputs, targets)
# print(loss)

class EqualizedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, beta=1.0):
        super(EqualizedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算易于分类的程度
        pt = torch.exp(-BCE_loss)
        
        # 计算 EFL
        focal_term = (1-pt).pow(self.gamma)
        eql_term = (1 - targets + targets * 1/self.beta) * targets
        EFL_loss = self.alpha * eql_term * focal_term * BCE_loss
        
        return EFL_loss.mean()
    
# # 示例用法
# criterion = EqualizedFocalLoss(alpha=0.25, gamma=2, beta=0.99)

# inputs = torch.randn(10, requires_grad=True)  # 模型的原始对数概率（logits）
# targets = torch.empty(10).random_(2)         # 真实的目标标签，随机生成的0或1

# loss = criterion(inputs, targets)
# print(loss)
