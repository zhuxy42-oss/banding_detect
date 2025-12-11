import meshio
import time
import numpy as np
from options.base_options import BaseOptions
from options.train_options import TrainOptions
from data import DataLoader
from torch_geometric.loader import DataLoader
from models import create_model
from util.writer import Writer
from util.util import make_dataset, smooth_map
from test import run_test
import torch
import torch.nn as nn
from models.networks import M2MRegressionLoss, M2MRegressionLoss1, M2MClassifyLoss, BinaryClassAccuracyLoss, EdgeRegressionGNN, MultiTaskEdgeGNN, EdgeClassificationGNN1, EdgeClassificationGNN2, EdgeClassificationGNNWithEdgeAttention, EdgeRegressionGNN, EdgeRegressionGNNWithEdgeAttention, CompiledPyGWrapper, ImprovedEdgeClassificationGNN, EdgeClassificationGNNWithKANClassifier, EdgeClassificationGNNWithHybridClassifier, EdgeClassificationGNNWithKANLayerAttention, EdgeClassificationGNNWithKANLayerAttentionClassify, EdgeClassificationGNNWithKANLayerAttentionClassify1, EdgeClassificationGNNWithKANLayerAttentionClassify2, EdgeClassificationGNNWithKANLayerAttentionClassify3, EdgeClassificationGNNWithKANLayerAttentionClassify4, EdgeRegressionGNNWithPureKAN, EdgeRegressionGNNWithPureKAN1, ThresholdClassificationLoss, EdgeClassificationGNNWithKANLayerAttentionClassify2_AblationAttention, EdgeClassificationGNNWithKANLayerAttentionClassify2_AblationClassifier1, EdgeClassificationGNNWithKANLayerAttentionClassify2_AblationConv2
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR, ExponentialLR, CosineAnnealingLR
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data.banding_data1 import gather_graph_normal2, gather_graph_normal3
import torch.nn.functional as F
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import pickle

def banding_accuracy(pred, label):
    batch = len(pred)
    total_correct = 0
    total_ones_match = 0
    total_zeros_match = 0
    total_ones = 0
    total_zeros = 0
    total_elements = 0
    
    for i in range(batch):
        predict = pred[i:i+1]
        target = label[i]
        
        edge_len = target.size(1)
        # Convert predictions to binary (0 or 1)
        predict_binary = (predict[:, 0:edge_len] > 0.5).float()
        
        # Calculate correct predictions
        correct = (predict_binary == target).float()
        total_correct += correct.sum().item()
        
        # Calculate where both are 1
        ones_match = (predict_binary * target).sum().item()
        total_ones_match += ones_match
        total_ones += target.sum().item()
        
        # Calculate where both are 0
        zeros_match = ((1 - predict_binary) * (1 - target)).sum().item()
        total_zeros_match += zeros_match
        total_zeros += (1 - target).sum().item()

        total_elements += edge_len
    
    # Calculate probabilities
    accuracy = total_correct / total_elements
    
    # Avoid division by zero
    ones_prob = total_ones_match / total_ones if total_ones > 0 else 0
    zeros_prob = total_zeros_match / total_zeros if total_zeros > 0 else 0

    return accuracy, ones_prob, zeros_prob
    
def test():
    graph1 = Data(
        x=torch.randn(10, 1),      # 10个节点，每个节点1个特征
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),  # 4条边
        edge_attr=torch.randn(4, 5)  # 4条边，每条边5个特征
    )

    graph2 = Data(
        x=torch.randn(20, 1),      # 20个节点，每个节点1个特征
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),  # 6条边
        edge_attr=torch.randn(6, 5)  # 6条边，每条边5个特征
    )

    # 创建数据集列表
    dataset = [graph1, graph2]

    # 创建数据加载器
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 遍历批次
    for batch in loader:
        print(f"批次中的节点总数: {batch.num_nodes}")
        print(f"批次中的边总数: {batch.num_edges}")
        print(f"点特征形状: {batch.x.shape}")  # 应为 [num_nodes, 1]
        print(f"边特征形状: {batch.edge_attr.shape}")  # 应为 [num_edges, 5]
        print(f"Batch向量: {batch.batch}")  # 指示每个节点属于哪个原始图

def lr_lambda(epoch):
    # 初始学习率
    initial_lr = 0.001
    # 目标最小学习率
    min_lr = 0.00001
    # 衰减率
    decay_factor = 0.99
    
    # 计算当前epoch的学习率
    current_lr = initial_lr * (decay_factor ** epoch)
    
    # 确保学习率不低于最小值
    return min(current_lr / initial_lr, min_lr / initial_lr)

def load_para():
    node_file = '/home/zhuxunyang/coding/banding_detect/para/node_stats.pkl'
    edge_file = '/home/zhuxunyang/coding/banding_detect/para/edge_stats.pkl'

    # 尝试加载已有参数
    if os.path.exists(node_file) and os.path.exists(edge_file):
        with open(node_file, 'rb') as f:
            node_mean, node_std = pickle.load(f)
        with open(edge_file, 'rb') as f:
            edge_mean, edge_std = pickle.load(f)
        return node_mean, node_std, edge_mean, edge_std

def train_regression(rito):

    torch.set_float32_matmul_precision('high')
    # 初始化选项
    # opt = TrainOptions().parse()
    print('Running Test')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset, _ = gather_graph_normal3("/home/zhuxunyang/coding/banding_detect/datasets/model_expanded", opt, device, 20, load_para())
    # val_dataset, _ = gather_graph_normal3("/home/zhuxunyang/coding/banding_detect/datasets/model_single_original", opt, device, 1, load_para())

    dataset, _ = gather_graph_normal2("/home/zhuxunyang/coding/banding_detect/datasets/model_expanded", device, 32, load_para(), rito)
    val_dataset, _ = gather_graph_normal2("/home/zhuxunyang/coding/banding_detect/datasets/model_single_original", device, 32, load_para(), rito)

    # model_gnn = EdgeRegressionGNN(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64,
    # ).to(device=device)

    model_gnn = EdgeRegressionGNNWithEdgeAttention(
        node_feat_dim=1,
        edge_feat_dim=6,
        hidden_dim=64,
    ).to(device=device)

    # model_gnn = EdgeRegressionGNNWithPureKAN1(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=32,
    # ).to(device=device)
    # warmup_batch = next(iter(dataset))
    # with torch.no_grad():
    #     # 预热编译
    #     model_gnn = torch.compile(model_gnn, dynamic=True, fullgraph=True)
    #     example_output = model_gnn(warmup_batch.x, warmup_batch.edge_index, warmup_batch.edge_attr)
    #     print("Compile success")

    MSE = nn.MSELoss()
    Classify_inner = M2MClassifyLoss()
    Regression_loss = nn.SmoothL1Loss()
    Classify = nn.BCEWithLogitsLoss()
    zero_one_accu = BinaryClassAccuracyLoss()
    this_loss = ThresholdClassificationLoss(threshold=rito, alpha=0, one_weight=torch.tensor([1.0], device=device))
    whole_epoch = 5000

    optimizer = optim.Adam(model_gnn.parameters(), lr=0.001)    #, weight_decay=0.0001

    # scheduler = MultiStepLR(optimizer, milestones=[opt.niter_decay], gamma=0.1)
    # scheduler = ExponentialLR(optimizer, gamma=0.999)
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=(whole_epoch // 3) * 2 ,
        eta_min=1e-5   # 最小学习率
    )

    threshold = 0.5

    train_losses1 = []
    train_losses2 = []
    classify_loss1 = []
    classify_loss2 = []
    val_train_losses1 = []
    val_train_losses2 = []
    val_classify_loss1 = []
    val_classify_loss2 = []
    epochs = []

    print(model_gnn)
    max_loss = 1000000
    

    for epoch in range(whole_epoch):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        loss_regression = 0
        loss_regression_inner = 0
        loss_classify = 0
        loss_classify_inner = 0
        loss_one_zero = 0
        val_loss_regression = 0
        val_loss_regression_inner = 0
        val_loss_classify = 0
        val_loss_classify_inner = 0
        val_loss_one_zero = 0
        

        for batch in dataset:
            out = model_gnn(batch)
            # out = model_gnn(batch.x, batch.edge_index, batch.edge_attr)
            label_regression = batch.y
            loss = this_loss(out, label_regression)
            loss_regression += loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_gnn.parameters(), max_norm=2.0)
            optimizer.step()
            optimizer.zero_grad()

        for val_batch in val_dataset:
            with torch.no_grad():
                val_label_regression = val_batch.y
                val_out = model_gnn(val_batch)
                # val_out = model_gnn(val_batch.x, val_batch.edge_index, val_batch.edge_attr)
                val_loss = this_loss(val_out, val_label_regression)
                val_loss_regression += val_loss

        if (loss_regression.item() < max_loss):
            max_loss = loss_regression.item()
            torch.save(model_gnn.state_dict(), "/home/zhuxunyang/coding/banding_detect/checkpoints/debug/GNNRegressionwithAttention.pth")

        print(f"Epoch {epoch}  MSE: {loss_regression.item():.8f}  Val MSE: {val_loss_regression.item():.8f} Learning rate: {optimizer.param_groups[0]['lr']}")
        scheduler.step()


def train_classify():
    torch.set_float32_matmul_precision('high')

    # 检查当前设置
    print(f"当前矩阵乘法精度设置: {torch.get_float32_matmul_precision()}")
    # 初始化选项
    # opt = TrainOptions().parse()
    # opt.serial_batches = True  # no shuffle
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset, _ = gather_graph_normal2("/home/zhuxunyang/coding/banding_detect/datasets/model_expanded1109", device, 32, load_para(), 1.05)
    val_dataset, _ = gather_graph_normal2("/home/zhuxunyang/coding/banding_detect/datasets/model_single_original1109", device, 32, load_para(), 1.05)
    # dataset, _ = gather_graph_normal2("/home/zhuxunyang/coding/banding_detect/datasets/model_expanded", opt, device, 20)
    # val_dataset, _ = gather_graph_normal2("/home/zhuxunyang/coding/banding_detect/datasets/model_single_original", opt, device, 1)

    

    model_gnn = EdgeClassificationGNNWithKANLayerAttentionClassify2(
        node_feat_dim=1,
        edge_feat_dim=7,
        hidden_dim=64
    ).to(device=device)
    
    # model_gnn = EdgeClassificationGNNWithKANLayerAttentionClassify2_AblationConv2(
    #     node_feat_dim=1,
    #     edge_feat_dim=7,
    #     hidden_dim=64
    # ).to(device=device)

    # model_gnn = EdgeClassificationGNNWithKANLayerAttentionClassify2_AblationAttention(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64
    # ).to(device=device)

    # model_gnn = EdgeClassificationGNNWithKANLayerAttentionClassify2_AblationClassifier1(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64
    # ).to(device=device)

    warmup_batch = next(iter(dataset))
    with torch.no_grad():
        # 预热编译
        model_gnn = torch.compile(model_gnn, dynamic=True, fullgraph=True)
        _ = model_gnn(warmup_batch.x, warmup_batch.edge_index, warmup_batch.edge_attr)
    
    print("模型编译完成，开始训练...")

    Classify = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([40.0], device=device))    #pos_weight=torch.tensor([4.0], device=device)
    BCE_ACCU = BinaryClassAccuracyLoss()

    optimizer = optim.Adam(model_gnn.parameters(), lr=0.001)    #, weight_decay=0.0001

    total_epoch = 6000

    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=(total_epoch // 6) ,
        eta_min=1e-4   # 最小学习率
    )

    # threshold = 0.5
    # train_losses1 = []
    # train_losses2 = []
    # classify_loss1 = []
    # classify_loss2 = []
    # val_train_losses1 = []
    # val_train_losses2 = []
    # val_classify_loss1 = []
    # val_classify_loss2 = []
    # epochs = []

    print(model_gnn)
    max_loss = 1000000000000

    print("Start train")

    for epoch in range(total_epoch):
        # loss_regression = 0
        loss_regression_inner = 0
        loss_classify = 0
        # loss_classify_inner = 0
        loss_one_zero = 0
        # val_loss_regression = 0
        # val_loss_regression_inner = 0
        val_loss_classify = 0
        # val_loss_classify_inner = 0
        val_loss_one_zero = 0
        
        for i, batch in enumerate(dataset):
            print(f'\rEpoch {epoch} Training: [{i+1}/{len(dataset)}]', end='', flush=True)
            label_regression = batch.y
            label_classify = batch.y1
            # out = model_gnn(batch)
            out = model_gnn(batch.x, batch.edge_index, batch.edge_attr)
            loss1 = Classify(out, label_classify)
            loss2 = BCE_ACCU(out, label_classify)
            loss_classify += loss1.item() # 只累加标量值
            loss_one_zero += loss2.item()
            loss = 1 * loss1 + 0 * loss2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_gnn.parameters(), max_norm=2.0)
            optimizer.step()
            optimizer.zero_grad()

        for i, val_batch in enumerate(val_dataset):
            with torch.no_grad():
                print(f'\rEpoch {epoch} Validating: [{i+1}/{len(val_dataset)}]', end='', flush=True)
                val_label_regression = val_batch.y
                val_label_classify = val_batch.y1
                # val_out = model_gnn(val_batch)
                val_out = model_gnn(val_batch.x, val_batch.edge_index, val_batch.edge_attr)
                val_loss1 = Classify(val_out, val_label_classify)
                val_loss2 = BCE_ACCU(val_out, val_label_classify)
                val_loss_classify += val_loss1.item()
                val_loss_one_zero += val_loss2.item()
        if (loss_classify < max_loss):
            max_loss = loss_classify
            torch.save(model_gnn.state_dict(), "/home/zhuxunyang/coding/banding_detect/checkpoints/debug/GNNKanAttentionClassify2_1205_Classify40_q1.05_new2_test.pth")

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr > 0.00001:
            scheduler.step()
        else:
            optimizer.param_groups[0]['lr'] = 0.00001  # 保持最低学习率

        print(f"Epoch {epoch}  BCE: {loss_classify:.8f} Banding loss: {loss_one_zero:.8f}  Val BCE: {val_loss_classify:.8f} Val Banding loss: {val_loss_one_zero:.8f} Learning rate: {optimizer.param_groups[0]['lr']}")

if __name__ == "__main__":
    train_classify()
    # train_regression(1.2)