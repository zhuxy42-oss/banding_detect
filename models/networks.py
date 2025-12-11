import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

import torch.nn.functional as F

from util.calculate_gradient import hausdorff_distance
import numpy as np
from torch_geometric.nn import GCNConv, GINEConv, BatchNorm, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, dropout_adj
from kan import KAN, KANLayer

###############################################################################
# Helper Functions
###############################################################################

class BinaryClassAccuracyLoss(nn.Module):
    def __init__(self, pos_weight=0.5, neg_weight=0.5, eps=1e-8):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.eps = eps
        
    def forward(self, pred, label):
        """
        Args:
            pred: 预测概率 [batch_size, edge_num] (0~1之间)
            target: 真实标签 [batch_size, edge_num] (0或1)
        Returns:
            组合损失值
        """
        total_correct = 0
        total_ones_match = 0
        total_zeros_match = 0
        total_ones = 0
        total_zeros = 0
        total_elements = 0
        batch = len(pred)
        # for i in range(batch):
        #     predict = pred[i:i+1]
        #     target = label[i]

        #     edge_len = 3000
        #     # Convert predictions to binary (0 or 1)
        #     predict_binary = (predict > 0.5).float()
            
        #     # Calculate correct predictions
        #     correct = (predict_binary == target).float()
        #     total_correct += correct.sum().item()
            
        #     # Calculate where both are 1
        #     ones_match = (predict_binary * target).sum().item()
        #     total_ones_match += ones_match
        #     total_ones += target.sum().item()
            
        #     # Calculate where both are 0
        #     zeros_match = ((1 - predict_binary) * (1 - target)).sum().item()
        #     total_zeros_match += zeros_match
        #     total_zeros += (1 - target).sum().item()

            # total_elements += edge_len
        
        predict_binary = (pred > 0.5).float()
            
        # Calculate correct predictions
        correct = (predict_binary == label).float()
        total_correct = correct.sum().item()
            
        # Calculate where both are 1
        ones_match = (predict_binary * label).sum().item()
        total_ones_match += ones_match
        total_ones = label.sum().item()
            
        # Calculate where both are 0
        zeros_match = ((1 - predict_binary) * (1 - label)).sum().item()
        total_zeros_match += zeros_match
        total_zeros = (1 - label).sum().item()
        # Avoid division by zero
        ones_prob = total_ones_match / total_ones if total_ones > 0 else 0
        zeros_prob = total_zeros_match / total_zeros if total_zeros > 0 else 0
            
        # 计算损失 (1 - 加权准确率)
        loss = torch.tensor([1 - (self.pos_weight * ones_prob + self.neg_weight * zeros_prob)], device='cuda').requires_grad_(True)
        return loss

class SimplificationLoss(nn.Module):
    def __init__(self):
        super(SimplificationLoss, self).__init__()
    
    def forward(self, origin_mesh, result_mesh, target_mesh):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss = torch.zeros(size=(len(origin_mesh), 1)).to(device)
        for i in range(len(origin_mesh)):
            # print(origin_mesh[i].vertices.requires_grad)
            # print(result_mesh[i].vertices.requires_grad)
            # print(target_mesh[i].vertices.requires_grad)
            loss1 = hausdorff_distance(result_mesh[i], target_mesh[i])
            # loss2 = target_mesh[i].get_gradinet() - origin_mesh[i].get_gradinet()
            loss[i] += (loss1)
        sum_loss = loss.mean()
        return sum_loss

class EdgeCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(EdgeCrossEntropyLoss, self).__init__()
    def forward(self, edge_prob, mesh, label):
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.L1Loss()
        Loss = []
        for i in range(len(mesh)):
            # true_prob = edge_prob[i, 0:mesh[i].edges_count]
            Loss.append(criterion(edge_prob, label[i]))
        return torch.mean(torch.stack(Loss))
class M2MRegressionLoss(nn.Module):
    def __init__(self):
        super(M2MRegressionLoss, self).__init__()
    def forward(self, pred, label):
        criterion = nn.MSELoss()
        Loss = []
        batch = pred.size(0)
        # pred = F.sigmoid(pred)
        for i in range(batch):
            edge_len = label[i].size(1)
            true_pred = pred[i, 0:edge_len].unsqueeze(0)
            true_label = label[i].requires_grad_(True)
            loss = criterion(true_pred, true_label)

            Loss.append(loss)
        Lossing = torch.stack(Loss)
        return torch.mean(Lossing).clone().detach().requires_grad_(True)
    
class M2MRegressionLoss1(nn.Module):
    def __init__(self):
        super(M2MRegressionLoss1, self).__init__()
    def forward(self, pred, label, mask):
        Loss = []
        batch = pred.size(0)
        for i in range(batch):
            edge_len = mask[i].size(1)
            true_pred = pred[i:i+1, 0:edge_len]
            true_label = label[i:i+1, 0:edge_len]
            lossing = torch.mean(torch.abs(true_label - true_pred) * (1 + 5 * mask[i]))    
            Loss.append(lossing)
        Lossing = torch.stack(Loss)
        return torch.mean(Lossing).clone().detach().requires_grad_(True)
    
class M2MClassifyLoss(nn.Module):
    def __init__(self):
        super(M2MClassifyLoss, self).__init__()
    def forward(self, pred, label):
        # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device='cuda'))    #pos_weight=torch.tensor([4.0], device=device)
        criterion = nn.BCELoss()
        Loss = []
        batch = len(pred)
        for i in range(batch):
            edge_len = label[i].size(1)
            true_pred = pred[i:i+1, 0:edge_len]
            true_label = label[i]
            loss = criterion(true_pred, true_label)
            Loss.append(loss)
        Lossing = torch.stack(Loss)
        return torch.mean(Lossing).clone().detach().requires_grad_(True)

class M2MRankLoss(nn.Module):
    def __init__(self, temperature=1.0, eps=1e-10):
        super(M2MRankLoss, self).__init__()
        self.temp = temperature  # 温度参数控制分布尖锐程度
        self.eps = eps
    def forward(self, pred, label):
        Loss = []
        for i in range(pred.size(0)):
            target_batch = label[i].requires_grad_(True)
            edge_len = target_batch.size(1)
            pred_batch = pred[i, 0:edge_len].unsqueeze(0)
            loss = -torch.sum(target_batch * torch.log(pred_batch + self.eps), dim=-1)
            Loss.append(loss)
        Lossing = torch.stack(Loss)
        return torch.mean(Lossing).clone().detach().requires_grad_(True)

class M2MRankLoss1(nn.Module):
    def __init__(self, margin=0.0001):
        super(M2MRankLoss1, self).__init__()
    def forward(self, pred, label):
        batch_size, _ = pred.shape
        loss = []
        for i in range(batch_size):
            edge_len = len(label[i])
            pred_batch = pred[i, 0:edge_len]
            target_batch = label[i].squeeze(0)

            target_diff = target_batch.unsqueeze(1) - target_batch.unsqueeze(0)
            pred_diff = pred_batch.unsqueeze(1) - pred_batch.unsqueeze(0)
            mask = (target_diff > 0).float()
            batch_loss = torch.mean(mask * torch.relu(pred_diff + self.margin))

            # 累加损失
            loss.append(batch_loss)

        return torch.mean(torch.stack(loss))


class EdgeRankLoss(nn.Module):
    def __init__(self):
        super(EdgeRankLoss, self).__init__()

    def forward(self, pred, target, margin):
        batch_size, edge = pred.shape

        # 初始化损失
        loss = []
        # 遍历每个 batch
        for i in range(batch_size):
            # 提取当前 batch 的预测值和目标值
            pred_batch = pred[i]
            target_batch = target[i]

            # 计算所有元素对的差异
            target_diff = target_batch.unsqueeze(1) - target_batch.unsqueeze(0)
            pred_diff = pred_batch.unsqueeze(1) - pred_batch.unsqueeze(0)

            # 仅保留 target[i] > target[j] 的元素对
            mask = (target_diff > 0).float()
            batch_loss = torch.mean(mask * torch.relu(pred_diff + margin))

            # 累加损失
            loss.append(batch_loss)
            

        return torch.mean(torch.stack(loss))

class ListNetLoss(nn.Module):
    def __init__(self, temperature=1.0, eps=1e-10):
        super().__init__()
        self.temp = temperature  # 温度参数控制分布尖锐程度
        self.eps = eps

    def forward(self, pred, target):
        """
        参数:
            pred: 预测分数 (batch_size, num_edges)
            target: 真实分数 (batch_size, num_edges)
        """
        # 生成概率分布（带温度参数的softmax）
        pred_p = pred / self.temp
        target_p = target / self.temp
        
        # 计算交叉熵损失（KL散度）
        loss = -torch.sum(target_p * torch.log(pred_p + self.eps), dim=-1)
        return torch.mean(loss)

class SpearmanLoss(nn.Module):
    def __init__(self, regularization='l2', reg_strength=1.0):
        super().__init__()
        self.reg = regularization
        self.reg_strength = reg_strength
        
    def differentiable_ranking(self, scores):
        """ 使用神经排序近似实现可微排序 """
        # 使用NeuralSort算法（需安装torchsort）
        from torchsort import soft_rank
        return soft_rank(scores, regularization=self.reg, regularization_strength=self.reg_strength)

    def forward(self, pred, target):
        """
        参数:
            pred: 预测分数 (batch_size, num_edges)
            target: 真实分数 (batch_size, num_edges)
        """
        # 计算可微排序
        pred_rank = self.differentiable_ranking(pred)
        target_rank = self.differentiable_ranking(target)
        
        # 计算Spearman相关系数
        pred_centered = pred_rank - pred_rank.mean(dim=-1, keepdim=True)
        target_centered = target_rank - target_rank.mean(dim=-1, keepdim=True)
        
        covariance = (pred_centered * target_centered).sum(dim=-1)
        pred_std = torch.sqrt((pred_centered**2).sum(dim=-1))
        target_std = torch.sqrt((target_centered**2).sum(dim=-1))
        
        spearman = covariance / (pred_std * target_std + 1e-8)
        
        # 将相关系数转换为损失（最大化相关系数 → 最小化 1 - rho）
        return 1.0 - torch.mean(spearman)

class ThresholdClassificationLoss(nn.Module):
    def __init__(self, threshold=1.2, alpha=0.5, temperature=1.0, one_weight = None):
        """
        Args:
            threshold: 分类阈值
            alpha: 回归损失和分类损失的权重比例
            temperature: 分类平滑温度参数
        """
        super().__init__()
        self.threshold = threshold
        self.alpha = alpha
        self.temperature = temperature
        self.one_weight = one_weight
        
    def forward(self, pred, target):
        """
        Args:
            pred: 模型预测值 [batch_size, 1] 或 [batch_size]
            target: 真实值 [batch_size, 1] 或 [batch_size]
        """
        pred = pred.squeeze()
        target = target.squeeze()
        
        # 1. 回归损失 - 保持原有的回归精度
        reg_loss = F.smooth_l1_loss(pred, target)
        
        # 2. 分类损失 - 基于阈值的分类任务
        # 将真实值转换为分类标签
        target_class = (target > self.threshold).float()
        
        # 使用sigmoid将预测值映射到0-1，然后应用温度参数
        pred_prob = torch.sigmoid((pred - self.threshold) / self.temperature)
        
        # 二元交叉熵损失
        cls_loss = F.binary_cross_entropy(input=pred_prob, target=target_class, weight=self.one_weight)
        
        # 3. 组合损失
        total_loss = self.alpha * reg_loss + (1 - self.alpha) * cls_loss
    
        return total_loss

def get_norm_layer(norm_type='instance', num_groups=1):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, affine=True, num_groups=num_groups)
    elif norm_type == 'none':
        norm_layer = NoNorm
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_norm_args(norm_layer, nfeats_list):
    if hasattr(norm_layer, '__name__') and norm_layer.__name__ == 'NoNorm':
        norm_args = [{'fake': True} for f in nfeats_list]
    elif norm_layer.func.__name__ == 'GroupNorm':
        norm_args = [{'num_channels': f} for f in nfeats_list]
    elif norm_layer.func.__name__ == 'BatchNorm2d':
        norm_args = [{'num_features': f} for f in nfeats_list]
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_layer.func.__name__)
    return norm_args

class NoNorm(nn.Module): #todo with abstractclass and pass
    def __init__(self, fake=True):
        self.fake = fake
        super(NoNorm, self).__init__()
    def forward(self, x):
        return x
    def __call__(self, x):
        return self.forward(x)

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def init_net(net, init_type, init_gain, gpu_ids):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)

    # net = net.to(torch.device('cpu'))

    if init_type != 'none':
        init_weights(net, init_type, init_gain)
    return net


def define_loss(opt):
    if opt.dataset_mode == 'classification':
        loss = torch.nn.CrossEntropyLoss()
    elif opt.dataset_mode == 'segmentation':
        loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    elif opt.dataset_mode == 'simplification':
        loss = SimplificationLoss()
    return loss





def reset_params(model): # todo replace with my init
    for i, m in enumerate(model.modules()):
        weight_init(m)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class HybridKANAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        # 第一层使用KANLayer捕获复杂模式
        self.kan_layer = KANLayer(
            in_dim=input_dim,
            out_dim=input_dim // 2
        )
        
        # 后续使用传统MLP进行细化
        self.mlp = nn.Sequential(
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.kan_layer(x)[0]
        return self.mlp(x)

class HybridKANAttention1(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        # 第一层使用KANLayer捕获复杂模式
        self.kan_layer = KANLayer(
            in_dim=input_dim,
            out_dim=input_dim // 2
        )
        self.layernorm1 = nn.LayerNorm(input_dim // 2)
        
        # 后续使用传统MLP进行细化
        self.mlp = nn.Sequential(
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.kan_layer(x)[0]
        return self.mlp(self.layernorm1(x))

class MultiHeadKANAttention(nn.Module):
    def __init__(self, input_dim, num_heads=3, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = input_dim // 2
        self.head_dim = head_dim
        
        # 多头KAN层
        self.kan_heads = nn.ModuleList([
            KANLayer(input_dim, head_dim) for _ in range(num_heads)
        ])
        
        # # 每个头单独的归一化层
        # self.head_norms = nn.ModuleList([
        #     nn.LayerNorm(head_dim) for _ in range(num_heads)
        # ])
        
        # 组合层前的归一化
        self.combined_norm = nn.LayerNorm(num_heads * head_dim)
        
        # 组合层
        self.kan_combine = KANLayer(num_heads * head_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        head_outputs = []
        for i, kan_head in enumerate(self.kan_heads):
            head_out = kan_head(x)[0]        # 每个头独立处理
            # head_out = self.head_norms[i](head_out)  # 头级别归一化
            head_out = torch.sigmoid(head_out)  # 激活函数
            head_out = self.dropout(head_out)
            head_outputs.append(head_out)
        
        # 拼接多头输出
        combined = torch.cat(head_outputs, dim=1)
        combined = self.combined_norm(combined)  # 组合特征归一化
        
        # 生成最终注意力分数
        attention_scores = torch.sigmoid(self.kan_combine(combined)[0])
        return attention_scores

class EdgeAwareAttention(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(EdgeAwareAttention, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # 注意力计算网络：同时考虑节点对和边特征
        self.attention_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.Tanh(),  # 或 ReLU/LeakyReLU
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, node_features, edge_index, edge_features):
        src, dst = edge_index
        
        # 源节点和目标节点特征
        h_src = node_features[src]  # [E, node_dim]
        h_dst = node_features[dst]  # [E, node_dim]
        
        # 拼接源节点、目标节点和边特征
        attention_input = torch.cat([h_src, h_dst, edge_features], dim=1)  # [E, 2*node_dim + edge_dim]
        
        # 计算注意力分数
        attention_scores = self.attention_net(attention_input)  # [E, 1]
        attention_weights = torch.sigmoid(attention_scores)  # 或 softmax
        
        return attention_weights.squeeze(-1)

class EdgeAwareGNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(EdgeAwareGNNLayer, self).__init__()
        
        # 节点特征变换
        self.node_transform = nn.Linear(node_dim, hidden_dim)
        
        # 边感知注意力机制
        self.edge_attention = EdgeAwareAttention(node_dim, edge_dim, hidden_dim)
        
        # 边特征变换（可选）
        self.edge_transform = nn.Linear(edge_dim, hidden_dim)
        
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        
        # 变换节点特征
        h_transformed = self.node_transform(x)  # [N, hidden_dim]
        
        # 计算边注意力权重
        attention_weights = self.edge_attention(x, edge_index, edge_attr)  # [E]
        
        # 变换边特征（可选）
        edge_transformed = self.edge_transform(edge_attr)  # [E, hidden_dim]
        
        # 消息传递：使用注意力权重加权聚合
        out = torch.zeros_like(h_transformed)
        
        for i in range(x.size(0)):
            # 找到指向节点i的所有边
            incoming_edges = (dst == i)
            if incoming_edges.sum() > 0:
                # 对应的源节点索引
                sources = src[incoming_edges]
                # 对应的注意力权重
                weights = attention_weights[incoming_edges]
                # 对应的变换后边特征
                edge_feats = edge_transformed[incoming_edges]
                
                # 加权聚合：节点特征 + 边特征
                weighted_messages = (h_transformed[sources] + edge_feats) * weights.unsqueeze(-1)
                out[i] = weighted_messages.sum(dim=0)
        
        return out
    


class EdgeClassificationGNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super(EdgeClassificationGNN, self).__init__()
        # 节点特征提取
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # 边分类器（输出单个logit值）
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)  # 输出1维logits
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 节点特征提取
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)  # [num_nodes, hidden_dim]

        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1) if edge_attr is not None else torch.cat([h_src, h_dst], dim=1)
        
        # 边分类logits
        edge_logits = self.edge_classifier(edge_repr)  # [num_edges, 1]
        return edge_logits  # 压缩为[num_edges]以适应BCEWithLogitsLoss

class EdgeClassificationGNN1(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super(EdgeClassificationGNN1, self).__init__()
        # 节点特征提取 - 使用GINEConv替代GCNConv
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        
        # 边分类器（输出单个logit值）
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)  # 输出1维logits
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # # 添加自环边（可选，但通常有助于GINEConv）
        # edge_index, edge_attr = add_self_loops(
        #     edge_index, 
        #     edge_attr=edge_attr,
        #     fill_value=0.0 if edge_attr is None else edge_attr.mean(dim=0),
        #     num_nodes=x.size(0)
        # )
        
        # 节点特征提取
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)  # [num_nodes, hidden_dim]

        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1) if edge_attr is not None else torch.cat([h_src, h_dst], dim=1)
        
        # 边分类logits
        edge_logits = self.edge_classifier(edge_repr)  # [num_edges, 1]
        return edge_logits


class EdgeClassificationGNN2(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super(EdgeClassificationGNN2, self).__init__()
        # 节点特征提取
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # 边特征提取
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ) if edge_feat_dim > 0 else None
        
        # 边分类器（输出单个logit值）
        self.edge_classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)  # 输出1维logits
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # # 验证输入
        # assert edge_index.max() < x.size(0), "Edge index contains invalid node indices"
        # assert x.size(1) == self.conv1.in_channels, "Input feature dimension mismatch"
        # if self.edge_encoder is not None and edge_attr is not None:
        #     assert edge_attr.size(1) == self.edge_feat_dim, "Edge feature dimension mismatch"

        # print(edge_index.max().item(), x.size(0))

        # 节点特征提取
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)  # [num_nodes, hidden_dim]

        edge_feature = self.edge_encoder(edge_attr)

        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_feature], dim=1) if edge_attr is not None else torch.cat([h_src, h_dst], dim=1)
        
        # 边分类logits
        edge_logits = self.edge_classifier(edge_repr)  # [num_edges, 1]
        return edge_logits  # 压缩为[num_edges]以适应BCEWithLogitsLoss

class EdgeClassificationGNNWithEdgeAttention(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super().__init__()
        
        # 保持原有的GINEConv层
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        
        # 边注意力机制
        self.edge_attention = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),  # 输出注意力分数
            nn.Sigmoid()  # 归一化到[0,1]
        )
        
        # 边分类器
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 节点特征提取
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        
        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1)
        
        # 计算边注意力权重
        attention_weights = self.edge_attention(edge_repr)
        
        # 应用注意力权重
        attended_edge_repr = edge_repr * attention_weights
        
        # 边分类
        edge_logits = self.edge_classifier(attended_edge_repr)
        return edge_logits

class EdgeClassificationGNNWithKANLayerAttention(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super().__init__()
        
        # GINEConv 层保持不变
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        
        # ==================== 修改部分开始 ====================
        # 使用 KANLayer 构建边注意力机制
        # 结构与你原来的 nn.Sequential 非常相似
        
        input_attention_dim = 2 * hidden_dim + edge_feat_dim
        
        self.edge_attention = HybridKANAttention(input_dim=input_attention_dim)
        # ==================== 修改部分结束 ====================
        
        # 边分类器 (保持不变)
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        # KANLayer 有自己的初始化，我们只初始化 Linear 层
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index, edge_attr):
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 节点特征提取
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        
        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1)
        
        # 计算边注意力权重
        attention_scores = self.edge_attention(edge_repr)
        attention_weights = attention_scores
        
        # 应用注意力权重
        attended_edge_repr = edge_repr * attention_weights
        
        # 边分类
        edge_logits = self.edge_classifier(attended_edge_repr)
        return edge_logits

class EdgeClassificationGNNWithKANLayerAttention1(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super().__init__()
        
        # GINEConv 层保持不变
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.3),  # 第一层dropout较小
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.3),  # 第一层dropout较小
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        
        # ==================== 修改部分开始 ====================
        # 使用 KANLayer 构建边注意力机制
        # 结构与你原来的 nn.Sequential 非常相似
        
        input_attention_dim = 2 * hidden_dim + edge_feat_dim
        
        self.edge_attention = HybridKANAttention1(input_dim=input_attention_dim)
        # ==================== 修改部分结束 ====================
        
        # 边分类器 (保持不变)
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim//2, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        # KANLayer 有自己的初始化，我们只初始化 Linear 层
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index, edge_attr):
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 节点特征提取
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        
        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1)
        
        # 计算边注意力权重
        attention_scores = self.edge_attention(edge_repr)
        attention_weights = attention_scores
        
        # 应用注意力权重
        attended_edge_repr = edge_repr * attention_weights
        
        # 边分类
        edge_logits = self.edge_classifier(attended_edge_repr)
        return edge_logits

class EdgeClassificationGNNWithKANLayerAttentionClassify(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super().__init__()
        
        # GINEConv 层保持不变
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        
        # ==================== 修改部分开始 ====================
        # 使用 KANLayer 构建边注意力机制
        # 结构与你原来的 nn.Sequential 非常相似
        
        input_attention_dim = 2 * hidden_dim + edge_feat_dim
        
        self.edge_attention = HybridKANAttention(input_dim=input_attention_dim)
        # ==================== 修改部分结束 ====================
        
        self.edge_classifier1 = KANLayer(2 * hidden_dim + edge_feat_dim, 2 * hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        # self.edge_classifier2 = KANLayer(hidden_dim, hidden_dim // 2)
        # self.edge_classifier3 = KANLayer(hidden_dim // 2, 1)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        # KANLayer 有自己的初始化，我们只初始化 Linear 层
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # def forward(self, data):
    def forward(self, x, edge_index, edge_attr):
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 节点特征提取
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        
        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1)
        
        # 计算边注意力权重
        attention_scores = self.edge_attention(edge_repr)
        attention_weights = attention_scores
        
        # 应用注意力权重
        attended_edge_repr = edge_repr * attention_weights
        
        # 边分类
        edge_logits1 = self.edge_classifier1(attended_edge_repr)[0]
        edge_logits1_dropout = self.dropout(edge_logits1)
        edge_logits = self.mlp(edge_logits1_dropout)
        return edge_logits

class EdgeClassificationGNNWithKANLayerAttentionClassify1(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super().__init__()
        
        # GINEConv 层保持不变
        # GINEConv 层添加更多正则化
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.3),  # 第一层dropout较小
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        
        # ==================== 修改部分开始 ====================
        # 使用 KANLayer 构建边注意力机制
        # 结构与你原来的 nn.Sequential 非常相似
        
        input_attention_dim = 2 * hidden_dim + edge_feat_dim
        
        self.edge_attention = HybridKANAttention1(input_dim=input_attention_dim)
        # ==================== 修改部分结束 ====================
        
        self.edge_classifier1 = KANLayer(2 * hidden_dim + edge_feat_dim, 2 * hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.kan_layernorm = nn.LayerNorm(2 * hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim//2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        # KANLayer 有自己的初始化，我们只初始化 Linear 层
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # def forward(self, data):
    def forward(self, x, edge_index, edge_attr):
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 节点特征提取
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        
        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1)
        
        # 计算边注意力权重
        attention_scores = self.edge_attention(edge_repr)
        attention_weights = attention_scores
        
        # 应用注意力权重
        attended_edge_repr = edge_repr * attention_weights
        
        # 边分类
        edge_logits1 = self.edge_classifier1(attended_edge_repr)[0]
        edge_logits1_dropout = self.dropout(edge_logits1)
        edge_logits = self.mlp(self.kan_layernorm(edge_logits1_dropout))
        return edge_logits
    
class EdgeClassificationGNNWithKANLayerAttentionClassify2(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super().__init__()
        
        # GINEConv 层
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        
        # ==================== 优化的 EdgeAwareAttention 集成 ====================
        self.edge_attention = EdgeAwareAttention(
            node_dim=hidden_dim, 
            edge_dim=edge_feat_dim, 
            hidden_dim=hidden_dim
        )
        
        # 注意力增强的边表示维度
        attention_enhanced_dim = 3 * hidden_dim + edge_feat_dim  # 源节点 + 目标节点 + 注意力特征 + 边特征
        
        # 使用 KANLayer 处理注意力增强的边表示
        self.edge_classifier1 = KANLayer(attention_enhanced_dim, 2 * hidden_dim)
        
        self.dropout = nn.Dropout(p=0.5)
        self.kan_layernorm = nn.LayerNorm(2 * hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim//2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index, edge_attr):
        # 节点特征提取
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        
        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        
        # ==================== 优化的注意力应用 ====================
        # 计算边注意力权重
        attention_weights = self.edge_attention(x, edge_index, edge_attr)
        
        # 创建注意力增强特征
        attention_enhanced = attention_weights.unsqueeze(-1) * (h_src + h_dst)
        
        # 构建完整的边表示：源节点 + 目标节点 + 注意力增强特征 + 原始边特征
        edge_repr = torch.cat([h_src, h_dst, attention_enhanced, edge_attr], dim=1)
        
        # 边分类
        edge_logits1 = self.edge_classifier1(edge_repr)[0]
        edge_logits1_dropout = self.dropout(edge_logits1)
        edge_logits = self.mlp(self.kan_layernorm(edge_logits1_dropout))
        return edge_logits

class EdgeClassificationGNNWithKANLayerAttentionClassify2_AblationAttention(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super().__init__()
        
        # GINEConv 层
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        
        
        # 注意力增强的边表示维度
        attention_enhanced_dim = 2 * hidden_dim + edge_feat_dim  # 源节点 + 目标节点 + 注意力特征 + 边特征
        
        # 使用 KANLayer 处理注意力增强的边表示
        self.edge_classifier1 = KANLayer(attention_enhanced_dim, 2 * hidden_dim)
        
        self.dropout = nn.Dropout(p=0.5)
        self.kan_layernorm = nn.LayerNorm(2 * hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim//2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index, edge_attr):
        # 节点特征提取
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        
        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        
        
        # 构建完整的边表示：源节点 + 目标节点 + 注意力增强特征 + 原始边特征
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1)
        
        # 边分类
        edge_logits1 = self.edge_classifier1(edge_repr)[0]
        edge_logits1_dropout = self.dropout(edge_logits1)
        edge_logits = self.mlp(self.kan_layernorm(edge_logits1_dropout))
        return edge_logits


class EdgeClassificationGNNWithKANLayerAttentionClassify2_AblationConv2(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super().__init__()
        
        # GINEConv 层
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        
        # ==================== 优化的 EdgeAwareAttention 集成 ====================
        self.edge_attention = EdgeAwareAttention(
            node_dim=hidden_dim, 
            edge_dim=edge_feat_dim, 
            hidden_dim=hidden_dim
        )
        
        # 注意力增强的边表示维度
        attention_enhanced_dim = 3 * hidden_dim + edge_feat_dim  # 源节点 + 目标节点 + 注意力特征 + 边特征
        
        # 使用 KANLayer 处理注意力增强的边表示
        self.edge_classifier1 = KANLayer(attention_enhanced_dim, 2 * hidden_dim)
        
        self.dropout = nn.Dropout(p=0.5)
        self.kan_layernorm = nn.LayerNorm(2 * hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim//2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index, edge_attr):
        # 节点特征提取
        x = self.conv1(x, edge_index, edge_attr)
        
        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        
        # ==================== 优化的注意力应用 ====================
        # 计算边注意力权重
        attention_weights = self.edge_attention(x, edge_index, edge_attr)
        
        # 创建注意力增强特征
        attention_enhanced = attention_weights.unsqueeze(-1) * (h_src + h_dst)
        
        # 构建完整的边表示：源节点 + 目标节点 + 注意力增强特征 + 原始边特征
        edge_repr = torch.cat([h_src, h_dst, attention_enhanced, edge_attr], dim=1)
        
        # 边分类
        edge_logits1 = self.edge_classifier1(edge_repr)[0]
        edge_logits1_dropout = self.dropout(edge_logits1)
        edge_logits = self.mlp(self.kan_layernorm(edge_logits1_dropout))
        return edge_logits

class EdgeClassificationGNNWithKANLayerAttentionClassify2_AblationClassifier1(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super().__init__()
        
        # GINEConv 层
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        
        # ==================== 优化的 EdgeAwareAttention 集成 ====================
        self.edge_attention = EdgeAwareAttention(
            node_dim=hidden_dim, 
            edge_dim=edge_feat_dim, 
            hidden_dim=hidden_dim
        )
        
        # 注意力增强的边表示维度
        attention_enhanced_dim = 3 * hidden_dim + edge_feat_dim  # 源节点 + 目标节点 + 注意力特征 + 边特征
        
        # 使用 KANLayer 处理注意力增强的边表示
        # self.edge_classifier1 = KANLayer(attention_enhanced_dim, 2 * hidden_dim)
        
        # self.dropout = nn.Dropout(p=0.5)
        # self.kan_layernorm = nn.LayerNorm(2 * hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(attention_enhanced_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim//2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index, edge_attr):
        # 节点特征提取
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        
        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        
        # ==================== 优化的注意力应用 ====================
        # 计算边注意力权重
        attention_weights = self.edge_attention(x, edge_index, edge_attr)
        
        # 创建注意力增强特征
        attention_enhanced = attention_weights.unsqueeze(-1) * (h_src + h_dst)
        
        # 构建完整的边表示：源节点 + 目标节点 + 注意力增强特征 + 原始边特征
        edge_repr = torch.cat([h_src, h_dst, attention_enhanced, edge_attr], dim=1)
        
        # 边分类
        # edge_logits1 = self.edge_classifier1(edge_repr)[0]
        # edge_logits1_dropout = self.dropout(edge_logits1)
        edge_logits = self.mlp(edge_repr)
        return edge_logits


class EdgeClassificationGNNWithKANLayerAttentionClassify3(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super().__init__()
        
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.Dropout(0.3),  # 第一层dropout较小
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        
        # 使用点边交互注意力
        input_attention_dim = 2 * hidden_dim + edge_feat_dim
        self.edge_attention = MultiHeadKANAttention(input_attention_dim)
        
        # 其余部分保持不变
        self.edge_classifier1 = KANLayer(2 * hidden_dim + edge_feat_dim, 2 * hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.kan_layernorm = nn.LayerNorm(2 * hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim//2, 1)
        )
        
    def forward(self, x, edge_index, edge_attr):
        # 节点特征提取
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        
        # 构建边表示
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1)
        
        # 使用点边交互注意力
        attention_scores = self.edge_attention(edge_repr)
        attended_edge_repr = edge_repr * attention_scores
        
        # 边分类
        edge_logits1 = self.edge_classifier1(attended_edge_repr)[0]
        edge_logits1_dropout = self.dropout(edge_logits1)
        edge_logits = self.mlp(self.kan_layernorm(edge_logits1_dropout))
        return edge_logits

class EdgeClassificationGNNWithKANLayerAttentionClassify4(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super().__init__()
        
        # GINEConv 层保持不变
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        
        # EdgeAwareAttention
        self.edge_attention = EdgeAwareAttention(
            node_dim=hidden_dim, 
            edge_dim=edge_feat_dim, 
            hidden_dim=hidden_dim
        )
        
        # 注意力增强的边表示维度
        attention_enhanced_dim = 3 * hidden_dim + edge_feat_dim
        
        # ==================== 纯KANLayer边分类器 ====================
        # 第一层KAN: 降维到隐藏维度
        self.kan_layer1 = KANLayer(attention_enhanced_dim, hidden_dim)
        
        # 第二层KAN: 进一步处理特征
        self.kan_layer2 = KANLayer(hidden_dim, hidden_dim // 2)
        
        # 第三层KAN: 输出层，输出1维logits
        self.kan_layer3 = KANLayer(hidden_dim // 2, 1)
        
        # 正则化层
        self.kan_layernorm1 = nn.LayerNorm(hidden_dim)
        self.kan_layernorm2 = nn.LayerNorm(hidden_dim // 2)
        
        # Dropout层
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.3)
        
        self._init_weights()

    def _init_weights(self):
        # 只初始化非KANLayer的层
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.LayerNorm)):
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index, edge_attr):
        # 节点特征提取
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        
        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        
        # 注意力机制
        attention_weights = self.edge_attention(x, edge_index, edge_attr)
        attention_enhanced = attention_weights.unsqueeze(-1) * (h_src + h_dst)
        
        # 构建完整的边表示
        edge_repr = torch.cat([h_src, h_dst, attention_enhanced, edge_attr], dim=1)
        
        # ==================== 纯KANLayer边分类 ====================
        # 第一层KAN + 正则化 + Dropout
        edge_feat1 = self.kan_layer1(edge_repr)[0]  # 获取KANLayer的输出
        edge_feat1 = self.kan_layernorm1(edge_feat1)
        edge_feat1 = F.tanh(edge_feat1)  # 添加激活函数
        edge_feat1 = self.dropout1(edge_feat1)
        
        # 第二层KAN + 正则化 + Dropout
        edge_feat2 = self.kan_layer2(edge_feat1)[0]
        edge_feat2 = self.kan_layernorm2(edge_feat2)
        edge_feat2 = F.tanh(edge_feat2)
        edge_feat2 = self.dropout2(edge_feat2)
        
        # 第三层KAN: 输出层
        edge_logits = self.kan_layer3(edge_feat2)[0]
        
        return edge_logits

class EdgeRegressionGNNWithPureKAN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super().__init__()
        
        # GINEConv 层保持不变
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        
        input_attention_dim = 2 * hidden_dim + edge_feat_dim
        self.edge_attention = HybridKANAttention1(input_dim=input_attention_dim)
        
        # 完全使用KANLayer替代MLP
        self.edge_classifier1 = KANLayer(2 * hidden_dim + edge_feat_dim, hidden_dim)
        self.edge_classifier2 = KANLayer(hidden_dim, hidden_dim//2)
        self.edge_classifier3 = KANLayer(hidden_dim//2, 1)
        
        self.dropout = nn.Dropout(p=0.5)
        self.kan_layernorm1 = nn.LayerNorm(hidden_dim)
        self.kan_layernorm2 = nn.LayerNorm(hidden_dim//2)
        
        # 输出偏移，确保输出至少为1
        # self.output_bias = nn.Parameter(torch.tensor(1.0))
        self.output_bias = torch.tensor([1.0], device='cuda')
        self._init_weights()

    def _init_weights(self):
        # KANLayer有自己的初始化，我们只初始化其他层
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1)
        
        attention_scores = self.edge_attention(edge_repr)
        attention_weights = attention_scores
        
        attended_edge_repr = edge_repr * attention_weights
        
        # 纯KANLayer前向传播
        edge_feat1 = self.edge_classifier1(attended_edge_repr)[0]
        edge_feat1_norm = self.kan_layernorm1(edge_feat1)
        edge_feat1_dropout = self.dropout(edge_feat1_norm)
        
        edge_feat2 = self.edge_classifier2(edge_feat1_dropout)[0]
        edge_feat2_norm = self.kan_layernorm2(edge_feat2)
        edge_feat2_dropout = self.dropout(edge_feat2_norm)
        
        raw_output = self.edge_classifier3(edge_feat2_dropout)[0]
        
        # 使用softplus确保输出为正，加上偏置确保至少为1
        final_output = F.relu(raw_output)
        final = final_output + self.output_bias
        
        
        return final

class EdgeRegressionGNNWithPureKAN1(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super().__init__()
        
        # GINEConv 层保持不变
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        
        input_attention_dim = 2 * hidden_dim + edge_feat_dim
        self.edge_attention = HybridKANAttention1(input_dim=input_attention_dim)
        
        # 回归器部分
        self.edge_regressor1 = KANLayer(2 * hidden_dim + edge_feat_dim, hidden_dim)
        # self.edge_regressor2 = KANLayer(hidden_dim, hidden_dim//2)
        self.edge_regressor3 = KANLayer(hidden_dim, 1)
        
        self.dropout = nn.Dropout(p=0.5)
        self.kan_layernorm1 = nn.LayerNorm(hidden_dim)
        # self.kan_layernorm2 = nn.LayerNorm(hidden_dim//2)
        
        # 针对1-3范围的输出处理
        # 使用softplus确保输出为正，加上偏置确保至少为1
        self.min_output_bias = nn.Parameter(torch.tensor(1.0))
        # 可选：添加输出缩放，帮助模型适应目标范围
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # 初始化输出参数
        nn.init.constant_(self.min_output_bias, 1.0)
        nn.init.constant_(self.output_scale, 1.0)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1)
        
        attention_scores = self.edge_attention(edge_repr)
        attention_weights = attention_scores
        
        attended_edge_repr = edge_repr * attention_weights
        
        # KAN层前向传播
        edge_feat1 = self.edge_regressor1(attended_edge_repr)[0]
        edge_feat1_norm = self.kan_layernorm1(edge_feat1)
        edge_feat1_dropout = self.dropout(edge_feat1_norm)
        
        # edge_feat2 = self.edge_regressor2(edge_feat1_dropout)[0]
        # edge_feat2_norm = self.kan_layernorm2(edge_feat2)
        # edge_feat2_dropout = self.dropout(edge_feat2_norm)
        
        raw_output = self.edge_regressor3(edge_feat1_dropout)[0]
        
        # 针对1-3范围的特殊输出处理
        # 方案1：softplus + 偏置（确保输出≥1）
        final_output = F.softplus(raw_output) + self.min_output_bias
        
        # 方案2：如果希望输出更集中在1-3范围，可以使用缩放后的sigmoid
        # final_output = torch.sigmoid(raw_output) * 2.0 + 1.0  # 映射到[1,3]
        
        # 方案3：带约束的ReLU（简单有效）
        # final_output = F.relu(raw_output) + self.min_output_bias
        
        return final_output

    def get_output_range_hint(self):
        """返回模型设计的输出范围提示"""
        return f"模型输出设计范围: ≥{self.min_output_bias.item():.2f} (可学习参数)"

class EdgeClassificationGNNWithKANClassifier(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, dropout_rate=0.5):
        super().__init__()
        
        # GINEConv层保持不变
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        
        # 边注意力机制保持不变
        self.edge_attention = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # ==================== 关键修改区域 ====================
        
        # 2. 定义 KAN 分类器的输入维度
        classifier_input_dim = 2 * hidden_dim + edge_feat_dim
        
        # (可选但推荐) 添加一个LayerNorm层来稳定KAN的输入
        self.layer_norm = nn.LayerNorm(classifier_input_dim)
        
        # 3. 用 KAN 替换原有的 nn.Sequential 分类器
        #    width: [输入维度, 隐藏层维度, 输出维度]
        #    grid: B-样条网格点数，越大越灵活，也越容易过拟合。5是一个很好的起点。
        #    k: B-样条的阶数，通常保持为3（三次样条）。
        self.edge_classifier_kan = KAN(
            width=[classifier_input_dim, hidden_dim // 2, 1], 
            grid=5, 
            k=3
        )
        
        # 4. Dropout层单独定义，将在forward中手动应用
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # ======================================================
        
        self._init_weights()

    def _init_weights(self):
        # KAN有自己的初始化方法，我们只需初始化Linear层
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 节点特征提取
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(x, p=self.dropout.p, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        
        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1)
        
        # 计算并应用边注意力权重
        attention_weights = self.edge_attention(edge_repr)
        attended_edge_repr = edge_repr * attention_weights
        
        # ==================== 关键修改区域 ====================
        
        # 5. 在输入到KAN之前应用LayerNorm和Dropout
        classifier_input = self.layer_norm(attended_edge_repr)
        classifier_input = self.dropout(classifier_input)
        
        # 6. 使用KAN进行边分类
        edge_logits = self.edge_classifier_kan(classifier_input)
        
        # ======================================================
        
        return edge_logits

class EdgeClassificationGNNWithHybridClassifier(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, dropout_rate=0.5):
        super().__init__()
        
        # GINEConv层
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.ReLU(),  # 改用ReLU加速
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        
        # 边注意力机制（简化）
        self.edge_attention = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        classifier_input_dim = 2 * hidden_dim + edge_feat_dim
        self.layer_norm = nn.LayerNorm(classifier_input_dim)
        
        self.feature_processor = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.edge_classifier_kan = KAN(
            width=[hidden_dim // 2, 8, 1],  # 很小的KAN
            grid=3,
            k=2
        )

        
        self.dropout = nn.Dropout(p=dropout_rate)
        self._init_weights()

    def _init_weights(self):
        # KAN有自己的初始化方法，我们只需初始化Linear层
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 节点特征提取
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(x, p=self.dropout.p, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        
        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1)
        
        # 注意力机制
        attention_weights = self.edge_attention(edge_repr)
        attended_edge_repr = edge_repr * attention_weights
        
        classifier_input = self.layer_norm(attended_edge_repr)
        classifier_input = self.dropout(classifier_input)

        # 混合模式：传统MLP + 小型KAN
        features = self.feature_processor(classifier_input)
        edge_logits = self.edge_classifier_kan(features)

        
        return edge_logits

class EdgeRegressionGNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super(EdgeRegressionGNN, self).__init__()
        # 节点特征提取 - 使用GINEConv替代GCNConv
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        
        # 边分类器（输出单个logit值）
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1),  # 输出1维logits
            nn.Sigmoid()
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # # 添加自环边（可选，但通常有助于GINEConv）
        # edge_index, edge_attr = add_self_loops(
        #     edge_index, 
        #     edge_attr=edge_attr,
        #     fill_value=0.0 if edge_attr is None else edge_attr.mean(dim=0),
        #     num_nodes=x.size(0)
        # )
        
        # 节点特征提取
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)  # [num_nodes, hidden_dim]

        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1) if edge_attr is not None else torch.cat([h_src, h_dst], dim=1)
        
        # 边分类logits
        edge_logits = self.edge_classifier(edge_repr)  # [num_edges, 1]
        return edge_logits

class EdgeRegressionGNNWithEdgeAttention(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super().__init__()
        
        # 保持原有的GINEConv层
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        
        # 边注意力机制
        self.edge_attention = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),  # 输出注意力分数
            nn.Sigmoid()  # 归一化到[0,1]
        )
        
        # 边回归器
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1),
            nn.ReLU()
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 节点特征提取
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        
        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1)
        
        # 计算边注意力权重
        attention_weights = self.edge_attention(edge_repr)
        
        # 应用注意力权重
        attended_edge_repr = edge_repr * attention_weights
        
        # 边分类
        edge_logits = self.edge_classifier(attended_edge_repr)
        return edge_logits + 1

    
class MultiTaskEdgeGNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super(MultiTaskEdgeGNN, self).__init__()
        # 共享的节点特征提取层
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # 边分类器（输出单个logit值）
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)  # 输出1维logits
        )
        
        # 边回归器（使用分类结果作为注意力）
        self.edge_regressor = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)  # 输出1维回归值
        )
        
        # 注意力权重生成器
        self.attention_generator = nn.Sequential(
            nn.Linear(1, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1),
            nn.ReLU()  # 输出0-1之间的注意力权重
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 1. 共享节点特征提取
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)  # [num_nodes, hidden_dim]

        # 2. 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1) if edge_attr is not None else torch.cat([h_src, h_dst], dim=1)
        
        # 3. 边分类任务
        edge_logits = self.edge_classifier(edge_repr)  # [num_edges, 1]
        
        # # 4. 基于分类结果生成注意力权重（仅对正样本关注）
        # with torch.no_grad():
        #     # 使用sigmoid将logits转换为概率
        #     class_probs = torch.sigmoid(edge_logits)
        #     # 生成注意力权重（正样本概率越高，注意力权重越大）
        #     attention_weights = self.attention_generator(class_probs)
        class_probs = torch.sigmoid(edge_logits)
        attention_weights = self.attention_generator(class_probs)
        
        # 5. 边回归任务（使用注意力权重加权）
        reg_values = self.edge_regressor(edge_repr)  # [num_edges, 1]
        weighted_reg_values = reg_values * attention_weights  # 应用注意力
        
        return {
            'classification': edge_logits,  # [num_edges, 1]
            'regression': weighted_reg_values,  # [num_edges, 1]
            'attention_weights': attention_weights  # [num_edges, 1]
        }

    def predict(self, data):
        """推理模式，返回分类概率和回归值"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(data)
            probs = torch.sigmoid(outputs['classification'])
            return {
                'probabilities': probs,
                'regression_values': outputs['regression'],
                'attention_weights': outputs['attention_weights']
            }

class ImprovedEdgeClassificationGNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=128, num_layers=3, 
                 dropout=0.3, num_attention_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 节点特征预处理
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
        # GNN层
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GINEConv(
                nn=nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Tanh(),
                    # nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim)
                ),
                edge_dim=edge_feat_dim
            ))
        
        # 改进的边注意力
        self.edge_attention = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 增强边分类器
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.Tanh(),
            # nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, 1)
        )
        
        # 残差连接相关的参数
        self.res_weights = nn.Parameter(torch.ones(num_layers))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 节点特征编码
        x = self.node_encoder(x)
        x_prev = x  # 用于残差连接
        
        # 多層GNN传播
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index, edge_attr)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # 带权重的残差连接
            if i > 0:
                alpha = torch.sigmoid(self.res_weights[i])
                x = alpha * x_new + (1 - alpha) * x_prev
            else:
                x = x_new
            
            x_prev = x_new
        
        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1)
        
        # 计算边注意力权重
        attention_weights = self.edge_attention(edge_repr)
        
        # 应用注意力权重
        attended_edge_repr = edge_repr * attention_weights
        
        # 边分类
        edge_logits = self.edge_classifier(attended_edge_repr)
        
        return edge_logits  # 返回注意力权重用于分析

class KAN_GCN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 保持原有的GINEConv层
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        
        # 边注意力机制
        self.edge_attention = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),  # 输出注意力分数
            nn.Sigmoid()  # 归一化到[0,1]
        )
        
        # KAN边分类器
        classifier_input_dim = hidden_dim + hidden_dim
        self.edge_classifier = KAN(width=[classifier_input_dim, hidden_dim//2, 1], grid=3, k=2)

    # ==================== 关键修改 ====================
    # forward方法现在只接收纯张量
    def forward(self, data):
        # 分批处理逻辑保持不变，因为它本身就在处理张量
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        
        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1)
        
        attention_weights = self.edge_attention(edge_repr)
        attended_edge_repr = edge_repr * attention_weights
        
        # 边分类
        edge_logits = self.edge_classifier(attended_edge_repr)
        
        return edge_logits

class ScalableKAN_Core(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=64, use_sparse_attention=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_sparse_attention = use_sparse_attention
        
        # KAN特征提取
        self.node_kan = KAN(width=[node_feat_dim, hidden_dim//2, hidden_dim], grid=3, k=2)
        self.edge_kan = KAN(width=[edge_feat_dim, hidden_dim//4, hidden_dim//2], grid=3, k=2)
        
        # GNN层
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
                nn.Linear(hidden_dim//2, hidden_dim)
            ),
            edge_dim=hidden_dim//2
        )
        
        # 注意力机制
        if use_sparse_attention:
            self.edge_attention = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim//2, hidden_dim//2), nn.ReLU(),
                nn.Linear(hidden_dim//2, 1), nn.Sigmoid()
            )
        else:
            self.edge_attention_kan = KAN(width=[hidden_dim * 2 + hidden_dim//2, hidden_dim//2, 1], grid=3, k=2)
        
        # KAN边分类器
        classifier_input_dim = hidden_dim + hidden_dim//2 if use_sparse_attention else hidden_dim * 2 + hidden_dim//2
        self.edge_classifier = KAN(width=[classifier_input_dim, hidden_dim//2, 1], grid=3, k=2)
        
        self.sigmoid = nn.Sigmoid()

    # ==================== 关键修改 ====================
    # forward方法现在只接收纯张量
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        # 分批处理逻辑保持不变，因为它本身就在处理张量
        x_kan = self.node_kan(x)
        edge_attr_kan = self.edge_kan(edge_attr)
        # GNN传播
        x_conv = self.conv1(x_kan, edge_index, edge_attr_kan)
        
        # 边表示
        src, dst = edge_index
        h_src = x_conv[src]
        
        if self.use_sparse_attention:
            edge_repr_for_attention = torch.cat([h_src, edge_attr_kan], dim=1)
        else:
            h_dst = x_conv[dst]
            edge_repr_for_attention = torch.cat([h_src, h_dst, edge_attr_kan], dim=1)
        
        # 注意力机制
        if self.use_sparse_attention:
            attention_weights = self.edge_attention(edge_repr_for_attention)
        else:
            # 注意：原始代码中这里缺少了sigmoid，对于KAN输出的logits，通常需要加一个
            attention_weights = self.sigmoid(self.edge_attention_kan(edge_repr_for_attention))
        
        attended_edge_repr = edge_repr_for_attention * attention_weights
        
        # 边分类
        edge_logits = self.edge_classifier(attended_edge_repr)
        
        return edge_logits
    
class CompiledPyGWrapper(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=64, use_sparse_attention=True):
        super().__init__()
        
        # 1. 实例化核心模型
        core_model = ScalableKAN_Core(
            node_feat_dim, edge_feat_dim, hidden_dim, use_sparse_attention
        )
        
        # 2. 在这里对核心模型进行编译！
        self.compiled_core_model = torch.compile(core_model)

    def forward(self, data):
        """
        这个forward方法与PyG兼容，它负责解包数据并调用编译后的核心模型。
        """
        # 3. 从PyG的data对象中解包出纯张量
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 如果你的模型还需要其他属性（如 data.batch），也在这里解包并传递
        # batch = data.batch
        
        # 4. 调用编译好的核心模型，传入纯张量
        return self.compiled_core_model(x, edge_index, edge_attr)

class PyGWrapper(nn.Module):
    def __init__(self, core_model):
        super().__init__()
        # 在包装器内部实例化核心模型
        self.core_model = core_model

    def forward(self, data):
        # 从 PyG 的 Data 对象中解包出张量
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 将解包后的张量传递给核心模型
        return self.core_model(x, edge_index, edge_attr)