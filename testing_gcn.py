import meshio
import time
import numpy as np
from options.base_options import BaseOptions
from options.test_options import TestOptions
from data import DataLoader
from torch_geometric.loader import DataLoader
from models import create_model
from util.writer import Writer
from util.util import make_dataset, smooth_map
from test import run_test
import torch
import torch.nn as nn
from models.networks import M2MRegressionLoss, M2MRegressionLoss1, M2MClassifyLoss, EdgeClassificationGNN2, EdgeClassificationGNN, EdgeRegressionGNN, MultiTaskEdgeGNN, EdgeClassificationGNN1, EdgeClassificationGNNWithEdgeAttention, EdgeRegressionGNN, EdgeRegressionGNNWithEdgeAttention, ImprovedEdgeClassificationGNN, EdgeClassificationGNNWithKANClassifier, EdgeClassificationGNNWithHybridClassifier, EdgeClassificationGNNWithKANLayerAttention, EdgeClassificationGNNWithKANLayerAttentionClassify, EdgeClassificationGNNWithKANLayerAttentionClassify1, EdgeClassificationGNNWithKANLayerAttentionClassify2, EdgeClassificationGNNWithKANLayerAttentionClassify4, EdgeRegressionGNNWithPureKAN, EdgeClassificationGNNWithKANLayerAttentionClassify2_AblationAttention, EdgeClassificationGNNWithKANLayerAttentionClassify2_AblationClassifier1, EdgeClassificationGNNWithKANLayerAttentionClassify2_AblationConv2
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
from models.layers.CustomMesh import barycentric_interpolation_batch, barycentric_interpolation_batch_vectorized
from torch_geometric.data import Data
from torch.serialization import add_safe_globals
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from data.banding_data1 import gather_graph_normal2
import subprocess
from models.layers import CustomMesh
from data.banding_data1 import load_or_compute_stats
import os
from collections import defaultdict
from scipy.spatial import KDTree
import trimesh
from training_gcn import load_para
from util.laplas_smooth import UnifiedLaplaceOptimizer, OneWayLaplaceOptimizer, ShrinkingLaplaceOptimizer, Gradinet
from util.xiao_nlp1 import smooth_mesh_sizing



def query_size(points, mesh):
    # 将 pyvista 网格转换为 trimesh 网格
    trimesh_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    t1 = time.time()
    index_point, distance, index_face = trimesh_mesh.nearest.on_surface(points)
    print(f"Cost {time.time() - t1}s")
    face_normal = mesh.face_normal
    # 计算点到三角面片所在平面的距离
    d = torch.sum(face_normal[index_face] * (points - index_point), dim=1)

    # 计算投影点
    projections = points - d.unsqueeze(1) * face_normal[index_face]

    return barycentric_interpolation_batch_vectorized(projections, mesh.faces[index_face], mesh)

def find_nearest_triangles(mesh1, mesh2):
    """
    为mesh1的每个三角面片找到mesh2中最近的三角面片
    
    参数:
    mesh1 - 源网格 (CustomMesh对象)
    mesh2 - 目标网格 (CustomMesh对象)
    kdtree - 可选的预构建KD树 (加速查询)
    
    返回:
    nearest_triangles - 每个mesh1面片对应的最近mesh2面片索引列表
    distances - 对应的距离列表
    """
    # 计算mesh1所有面片的中心点
    mesh1_vertices = mesh1.vertices.numpy()
    mesh1_faces = mesh1.faces.numpy()
    mesh1_centroids = np.mean(mesh1_vertices[mesh1_faces], axis=1)
    
    # 计算mesh2所有面片的中心点和法向量
    mesh2_vertices = mesh2.vertices.numpy()
    mesh2_faces = mesh2.faces.numpy()
    mesh2_centroids = np.mean(mesh2_vertices[mesh2_faces], axis=1)
    
    kdtree = KDTree(mesh2_centroids)
    
    # 查询每个mesh1中心点的最近mesh2中心点
    _, nearest_indices = kdtree.query(mesh1_centroids)
    
    for i in range(len(nearest_indices)):
        if i == 605:
            print(i, nearest_indices[i])
        mesh1.surface_id[i] = mesh2.surface_id[nearest_indices[i]]

def banding_accuracy(pred, label, i):
    batch = len(pred)
    total_correct = 0
    total_ones_match = 0
    total_zeros_match = 0
    total_ones = 0
    total_zeros = 0
    total_elements = 0
    
    # Convert predictions to binary (0 or 1)
    predict_binary = (pred > 0.5).float()
        
    # Calculate correct predictions
    correct = (predict_binary == label)
    total_correct = correct.sum().item()
        
    # Calculate where both are 1
    ones_match = (predict_binary * label).sum().item()
    total_ones_match = ones_match
    total_ones = label.sum().item()
        
    # Calculate where both are 0
    zeros_match = ((1 - predict_binary) * (1 - label)).sum().item()
    total_zeros_match = zeros_match
    total_zeros = (1 - label).sum().item()

    total_elements = len(label)
    
    # Calculate probabilities
    accuracy = total_correct / total_elements
    
    # Avoid division by zero
    ones_prob = total_ones_match / total_ones if total_ones > 0 else 0
    zeros_prob = total_zeros_match / total_zeros if total_zeros > 0 else 0

    # 调整输出格式，确保对齐
    # 使用单个print语句，通过制表符(\t)和固定宽度格式化确保对齐
    print(f"{i}\t" f"Accuracy: {accuracy:8.5f} = {int(total_correct):6d} / {int(total_elements):6d}\t"
          f"Ones Prob: {ones_prob:8.5f} = {int(total_ones_match):6d} / {int(total_ones):6d}\t"
          f"Zeros Prob: {zeros_prob:8.5f} = {int(total_zeros_match):6d} / {int(total_zeros):6d}")

    return total_ones_match, total_zeros_match, total_ones, total_zeros
    
# def gather_graph(path, opt, device, batch_size):
#     vtk_path, pt_path = make_dataset(path)
#     train_len = len(vtk_path)
#     graph_list = []
#     for i in range(train_len):
#         mesh = Mesh(file=vtk_path[i], opt=opt, hold_history=True, export_folder=opt.export_folder)
#         target = smooth_map(torch.load(pt_path[i], weights_only=True)).unsqueeze(0).requires_grad_(True).to(device).float() 
#         edge_features = mesh.extract_features()
#         data = Data(x=torch.from_numpy(mesh.size_value).float(), edge_index=torch.from_numpy(mesh.edges).transpose(1, 0).long(), edge_attr=torch.from_numpy(edge_features).transpose(1, 0).float(), pos=torch.from_numpy(mesh.vs)).to(device)
#         data.y = target.transpose(1, 0)
#         data.y1 = (target.transpose(1, 0) > 0.5).float()
#         graph_list.append(data)
#     loader = DataLoader(graph_list, batch_size=batch_size)
#     return loader

# def create_example(vtk_path, opt, device, norm_params=None):
#     mesh = Mesh(file=vtk_path, opt=opt, hold_history=True, export_folder=opt.export_folder)
#     all_node_features, all_edge_features = [], []
#     all_node_features.append(mesh.size_value)
#     all_edge_features.append(mesh.extract_features())
#     node_mean, node_std, edge_mean, edge_std = load_or_compute_stats(np.concatenate(all_node_features, axis=0), np.concatenate(all_edge_features, axis=1))
#     edge_features = mesh.extract_features()
#     GEO = vtk_path.split('/')[-1].split('_')[0]
#     edge_len = len(mesh.edges)

#     # 归一化（假设使用全局参数）
#     norm_node = (mesh.size_value - node_mean) / node_std
#     norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
    
#     # 检查并转换顶点数据的字节序
#     if mesh.vs.dtype.byteorder != '=':
#         mesh.vs = mesh.vs.byteswap().newbyteorder()

#     data = Data(
#         x=torch.from_numpy(norm_node).float(),
#         edge_index=torch.from_numpy(mesh.edges).T.long(),
#         edge_attr=torch.from_numpy(norm_edge).T.float(),
#         pos=torch.from_numpy(mesh.vs),
#         mesh = CustomMesh.CustomMesh.from_vtk(vtk_path),
#         geo = GEO
#     ).to(device)
#     return DataLoader([data], batch_size=1, shuffle=True), (node_mean, node_std, edge_mean, edge_std)

def create_example1(vtk_path, device, norm_params=None):
    mesh = CustomMesh.CustomMesh.from_vtk(vtk_path)
    all_node_features, all_edge_features = [], []
    all_node_features.append(mesh.sizing_values)
    edge_features = mesh.compute_edge_features()
    all_edge_features.append(edge_features)
    node_mean, node_std, edge_mean, edge_std = load_or_compute_stats(np.concatenate(all_node_features, axis=0), np.concatenate(all_edge_features, axis=1))
    
    GEO = vtk_path.split('/')[-1].split('_')[0]
    edge_len = len(mesh.edges)

    # 归一化（假设使用全局参数）
    norm_node = (mesh.sizing_values - node_mean) / node_std
    norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)

    data = Data(
        x=norm_node.float(),
        edge_index=mesh.edges.T.long(),
        edge_attr=norm_edge.T.float(),
        pos=mesh.vertices,
        mesh = mesh,
        geo = GEO
    ).to(device)
    return DataLoader([data], batch_size=1, shuffle=True), (node_mean, node_std, edge_mean, edge_std)


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

def test_classify():
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # val_dataset, _ = gather_graph_normal("/home/zhuxunyang/coding/banding_detect/datasets/model0501", opt, device, 1)
    val_dataset, _ = gather_graph_normal2("/home/zhuxunyang/coding/banding_detect/datasets/model_single_original", device, 1, load_para(), 1.3)
    # add_safe_globals([EdgeClassificationGNN, GCNConv])
    # model_gnn = EdgeClassificationGNN1(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64
    # ).to(device=device)
    # model_gnn = EdgeClassificationGNNWithEdgeAttention(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64
    # ).to(device=device)

    # model_gnn = EdgeClassificationGNNWithKANClassifier(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64
    # ).to(device=device)

    # model_gnn = ImprovedEdgeClassificationGNN(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64
    # ).to(device=device)

    # model_gnn = EdgeClassificationGNNWithHybridClassifier(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64
    # ).to(device=device)

    # model_gnn = EdgeClassificationGNNWithKANLayerAttention(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64
    # ).to(device=device)

    model_gnn = EdgeClassificationGNNWithKANLayerAttentionClassify(
        node_feat_dim=1,
        edge_feat_dim=6,
        hidden_dim=64 
    ).to(device=device)

    model_gnn = torch.compile(model_gnn, dynamic=True, fullgraph=False)

    checkpoint = torch.load(
        # "/home/zhuxunyang/coding/banding_detect/checkpoints/debug/GNN_new_Classify1_q1.02_new1.pth",
        "/home/zhuxunyang/coding/banding_detect/checkpoints/debug/GNNKanAttentionClassify_1027_Classify8_q1.1.pth",
        # "/home/zhuxunyang/coding/banding_detect/checkpoints/debug/GNNeAttentionClassifyHyperKAN_1024_Classify8_q1.3.pth",
        weights_only=True
    )

    model_gnn.load_state_dict(checkpoint)
    MSE = nn.MSELoss()
    L1 = nn.L1Loss()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sum_pred_ones = 0
    sum_pred_zeros = 0
    sum_edges = 0
    sum_label_ones = 0
    sum_label_zeros = 0
    i = 0
    for val_batch in val_dataset:
        with torch.no_grad():
            val_label_regression = val_batch.y
            val_label_classify = val_batch.y1
            # val_out = F.sigmoid(model_gnn(val_batch))
            val_out = F.sigmoid(model_gnn(val_batch.x, val_batch.edge_index, val_batch.edge_attr))
            classify_result = (val_out > 0.5).float()
            ones_correct, zeros_correct, ones_label, zeros_label = banding_accuracy(classify_result, val_label_classify, i)
            sum_pred_ones += ones_correct
            sum_pred_zeros += zeros_correct
            sum_edges += ones_label
            sum_edges += zeros_label
            sum_label_ones += ones_label
            sum_label_zeros += zeros_label
            i += 1

    print("Total accu:", (sum_pred_ones + sum_pred_zeros) / sum_edges, "Ones accu:", sum_pred_ones / sum_label_ones,"Zeros accu:", sum_pred_zeros / sum_label_zeros)

# def test_example(path):
#     print('Running Test')
#     opt = TestOptions().parse()
#     opt.serial_batches = True  # no shuffle
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     val_dataset, _ = gather_graph_normal(path, opt, device, 1)
#     add_safe_globals([EdgeClassificationGNN, GCNConv])
#     model_gnn = EdgeClassificationGNN1(
#         node_feat_dim=1,
#         edge_feat_dim=5,
#         hidden_dim=64
#     ).to(device=device)
#     checkpoint = torch.load(
#         "/home/zhuxunyang/coding/banding_detect/checkpoints/debug/GNN1_Classify8_q1.2.pth",
#         weights_only=False
#     )
#     model_gnn.load_state_dict(checkpoint)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     val_batch = val_dataset.dataset[0]
#     sum_pred_ones = 0
#     sum_pred_zeros = 0
#     sum_edges = 0
#     sum_label_ones = 0
#     sum_label_zeros = 0
#     i = 0
#     with torch.no_grad():
#         val_out = F.sigmoid(model_gnn(val_batch)).T
#         classify_result = (val_out > 0.5).float()
#         val_label_classify = val_batch.y1
#         geo_num = val_batch.geo
#         ones_correct, zeros_correct, ones_label, zeros_label = banding_accuracy(classify_result.T, val_label_classify, 0)
#         sum_pred_ones = ones_correct
#         sum_pred_zeros = zeros_correct
#         sum_edges = ones_label + zeros_label
#         sum_label_ones = ones_label
#         sum_label_zeros = zeros_label

#         result_mesh = val_batch.mesh.solve_banding(classify_result)
#         save_path1 = "/home/zhuxunyang/coding/banding_detect/result.vtk"
#         save_path2 = f"/home/zhuxunyang/coding/banding_detect/result_bkgm_{geo_num}.vtk"
#         geo_path = f"/home/zhuxunyang/coding/banding_detect/datasets/banding_dataset/test/{geo_num}/{geo_num}.step"
#         result_mesh.writeVTK(save_path1)

#         command = ['./test_sizefield2', save_path1, save_path2, '1.2', '0', '0.1', geo_path]
#         try:
#             result = subprocess.run(command, check=True, text=True, capture_output=True)
#         except subprocess.CalledProcessError as e:
#             print(e.stderr)
#             return None

#         print("Total accu:", (sum_pred_ones + sum_pred_zeros) / sum_edges, "Ones accu:", sum_pred_ones / sum_label_ones,"Zeros accu:", sum_pred_zeros / sum_label_zeros)
        
# def test_example1(path):
#     total_time = defaultdict(float)
#     start_time = time.time()
#     print('Running Test')
#     opt = TestOptions().parse()
#     opt.serial_batches = True  # no shuffle
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # val_dataset, norm_params = gather_graph_normal(path, opt, device, 1)
#     val_dataset, norm_params = create_example(path, opt, device)
#     add_safe_globals([EdgeClassificationGNN, GCNConv])
#     model_gnn = EdgeClassificationGNN1(
#         node_feat_dim=1,
#         edge_feat_dim=5,
#         hidden_dim=64
#     ).to(device=device)
#     checkpoint = torch.load(
#         "/home/zhuxunyang/coding/banding_detect/checkpoints/debug/GNN1_Classify1_q1.2_new.pth",
#         weights_only=False
#     )

#     # model_gnn = EdgeClassificationGNN2(
#     #     node_feat_dim=1,
#     #     edge_feat_dim=5,
#     #     hidden_dim=64
#     # ).to(device=device)
#     # checkpoint = torch.load(
#     #     "/home/zhuxunyang/coding/banding_detect/checkpoints/debug/GNN2_Classify1_q1.2_new.pth",
#     #     weights_only=False
#     # )

#     model_gnn.load_state_dict(checkpoint)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     val_batch = val_dataset.dataset[0]
#     sum_pred_ones = 0
#     sum_pred_zeros = 0
#     sum_edges = 0
#     sum_label_ones = 0
#     sum_label_zeros = 0
#     i = 0
#     Flag = True
#     flag = 0
#     step = 0
#     with torch.no_grad():
#         while Flag:
#             start_time = time.time()
#             val_out = F.sigmoid(model_gnn(val_batch)).T
#             classify_result = (val_out > 0.5).float()
#             geo_num = val_batch.geo
#             num_before = sum(val_batch.mesh.num_mesh())
            
#             loop_time = time.time() - start_time
#             save_path1 = f"/home/zhuxunyang/coding/banding_detect/middle/refined_{step}.vtk"
#             save_path2 = f"/home/zhuxunyang/coding/banding_detect/middle/refined_sf_{step}.vtk"
#             save_path3 = f"/home/zhuxunyang/coding/banding_detect/middle/{step}_visual.vtk"
#             save_path4 = f"/home/zhuxunyang/coding/banding_detect/middle/refined_original_{step}.vtk"
#             geo_path = f"/home/zhuxunyang/coding/banding_detect/datasets/banding_dataset/test/{geo_num}/{geo_num}.step"
#             print("Predict-time:", loop_time, " s")
#             # val_batch.mesh.visual_edge(classify_result[0], save_path3)
#             result_mesh = val_batch.mesh.solve_banding(classify_result)
            
#             result_mesh.writeVTK(save_path1)

#             command1 = ['./remesh', '--input', save_path1, '--eps', '1e-3', '--envelope-dis', '5e-4', '--output', save_path4,'--split-num', '0', '--collapse-num', '0']
#             try:
#                 result1 = subprocess.run(command1, check=True, text=True, capture_output=True)
#                 print("remesh success")
#             except subprocess.CalledProcessError as e:
#                 print("remesh fail")
#                 print(e.stderr)

#             command = ['./test_sizefield_noIDmap', save_path4, save_path2, '1.2', '0', '0.1', geo_path]
#             try:
#                 result = subprocess.run(command, check=True, text=True, capture_output=True)
#             except subprocess.CalledProcessError as e:
#                 # print(result.stdout)
#                 return None
            
#             step_time = time.time()
#             print(f"{step} Cost {step_time - start_time:.2f}s")
            
#             original_mesh = CustomMesh.CustomMesh.from_vtk(save_path2)
#             num_after = sum(original_mesh.num_mesh())
#             model_mesh = Mesh(file=save_path2, opt=opt)

#             r = num_before / num_after

#             if r < 1.02 and r >= 1:
#                 flag += 1
#                 print(save_path2, r)
#             elif r < 1:
#                 flag += 1
#                 print(f"The rito <= {r}, this refinement is not accept")
#                 os.remove(save_path1)
#                 os.remove(save_path2)
#                 continue
#             else:
#                 flag = 0
#                 print(save_path2, r)
            
#             if flag >= 3:
#                 Flag = False

#             # 提取新网格的特征并归一化
#             node_mean, node_std, edge_mean, edge_std = norm_params
#             edge_features = model_mesh.extract_features()
#             norm_node = (model_mesh.size_value - node_mean) / node_std
#             norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
#             val_batch.x = torch.from_numpy(norm_node).float().to(device)
#             val_batch.edge_index = torch.from_numpy(model_mesh.edges).T.long().to(device)
#             val_batch.edge_attr = torch.from_numpy(norm_edge).T.float().to(device)
#             val_batch.pos = torch.from_numpy(model_mesh.vs).to(device)
#             val_batch.mesh = original_mesh  # 注意保持 batch 结构
            
#             step += 1
#     end_time = time.time()
#     print(f"Cost {end_time - start_time:.2f}s")

def test_example2(path):
    total_time = defaultdict(float)
    begin_time = time.time()
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # val_dataset, norm_params = gather_graph_normal(path, opt, device, 1)
    # val_dataset, norm_params = create_example(path, opt, device)
    val_dataset, norm_params = create_example1(path, device)
    add_safe_globals([EdgeClassificationGNN, GCNConv])
    model_gnn = EdgeClassificationGNN1(
        node_feat_dim=1,
        edge_feat_dim=5,
        hidden_dim=64
    ).to(device=device)
    model_gnn1 = EdgeClassificationGNN1(
        node_feat_dim=1,
        edge_feat_dim=5,
        hidden_dim=64
    ).to(device=device)
    checkpoint = torch.load(
        "/home/zhuxunyang/coding/banding_detect/checkpoints/debug/GNN1_Classify1_q1.05_new1.pth",
        weights_only=False
    )
    checkpoint1 = torch.load(
        "/home/zhuxunyang/coding/banding_detect/checkpoints/debug/GNN1_Classify1_q1.1_new1.pth",
        weights_only=False
    )

    model_gnn.load_state_dict(checkpoint)
    model_gnn1.load_state_dict(checkpoint1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_batch = val_dataset.dataset[0]
    sum_pred_ones = 0
    sum_pred_zeros = 0
    sum_edges = 0
    sum_label_ones = 0
    sum_label_zeros = 0
    i = 0
    Flag = True
    flag = 0
    step = 0
    begin_cell = len(val_batch.mesh.faces)
    end_mesh = None

    step1_start_time = time.time()

    with torch.no_grad():
        while Flag:
            start_time = time.time()
            if len(val_batch.mesh.faces) > begin_cell * 1.2:
                val_out = F.sigmoid(model_gnn1(val_batch)).T
            else:
                val_out = F.sigmoid(model_gnn(val_batch)).T
            classify_result = (val_out > 0.5).float().to("cpu")
            geo_num = val_batch.geo
            num_before = sum(val_batch.mesh.num_mesh())
            
            loop_time = time.time() - start_time
            mesh = val_batch.mesh
            print("Predict-time:", loop_time, " s")
            save_path3 = f"/home/zhuxunyang/coding/banding_detect/middle/{step}_visual.vtk"
            val_batch.mesh.visual_edge(classify_result[0], save_path3)
            result_mesh = mesh.solve_banding2(classify_result)

            if len(result_mesh.faces) > begin_cell * 5:
                # end_mesh = save_path2
                print("too large")
                break

            save_path1 = f"/home/zhuxunyang/coding/banding_detect/middle/refined_{step}.vtk"
            save_path2 = f"/home/zhuxunyang/coding/banding_detect/middle/refined_sf_{step}.vtk"
            
            save_path4 = f"/home/zhuxunyang/coding/banding_detect/middle/refined_original_{step}.vtk"
            geo_path = f"/home/zhuxunyang/coding/banding_detect/datasets/banding_dataset/test/{geo_num}/{geo_num}.step"
            
            result_mesh.writeVTK(save_path1)

            command1 = ['./remesh', '--input', save_path1, '--eps', '1e-3', '--envelope-dis', '5e-4', '--output', save_path4,'--split-num', '0', '--collapse-num', '0', '--feature-angle', '1']
            try:
                result1 = subprocess.run(command1, check=True, text=True, capture_output=True)
                print("remesh success")
            except subprocess.CalledProcessError as e:
                print("remesh fail")
                print(e.stderr)

            # this_mesh = CustomMesh.CustomMesh.from_vtk(save_path4)
            # find_nearest_triangles(this_mesh, result_mesh)
            # this_mesh.writeVTK(save_path4)

            # command = ['./test_sizefield_noIDmap', save_path4, save_path2, '1.2', '0', '0.1', geo_path]
            command = ['./test_sizefield_nosmooth', save_path4, save_path2, '1.2', '0', '0.1', geo_path]
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                # print(result.stdout)
                return None
            
            step_time = time.time()
            print(f"{step} Cost {step_time - start_time:.2f}s")
            
            original_mesh = CustomMesh.CustomMesh.from_vtk(save_path2)
            num_after = sum(original_mesh.num_mesh())
            # model_mesh = Mesh(file=save_path2, opt=opt)

            r = num_before / num_after

            if r < 1.05 and r >= 1:
                flag += 1
                end_mesh = save_path2
                print(save_path2, r)
            elif r < 1:
                flag += 1
                print(f"The rito <= {r}, this refinement is not accept")
                os.remove(save_path1)
                os.remove(save_path2)
                continue
                
            else:
                flag = 0
                end_mesh = save_path2
                print(save_path2, r)

            if flag >= 5:
                Flag = False
                break

            # 提取新网格的特征并归一化
            node_mean, node_std, edge_mean, edge_std = norm_params
            edge_features = original_mesh.compute_edge_features()
            norm_node = (original_mesh.sizing_values - node_mean) / node_std
            norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
            val_batch.x = norm_node.float().to(device)
            val_batch.edge_index = original_mesh.edges.T.long().to(device)
            val_batch.edge_attr = norm_edge.T.float().to(device)
            val_batch.pos = original_mesh.vertices.to(device)
            val_batch.mesh = original_mesh  # 注意保持 batch 结构
            
            step += 1
    
    step1_end_time = time.time() - step1_start_time
    print("Step1 time:", step1_end_time)

    # step2_start_time = time.time()
    
    # print("We will continue process", end_mesh)
    # mesh = CustomMesh.CustomMesh.from_vtk(end_mesh)

    # result = []
    # Flag = True

    # num_before = sum(mesh.num_mesh())
    # size = []
    # print("Size querying")
    # process = subprocess.Popen(
    #     ['./test_querysize', end_mesh, geo_path, '0.1', '1.2'],
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE
    # )
    # stdout, stderr = process.communicate()
    # print("Size query complete")
    # if stderr:
    #     print("Error:", stderr.decode())
    #     return None
    # edge_num = len(mesh.edges)
    # output_lines = stdout.decode().splitlines()
    # for i in range(edge_num, 0, -1):
    #     size.append(float(output_lines[-i]))
    # size = torch.tensor(size)
    # assert len(size) == edge_num
    # # for i, edge in enumerate(mesh.edges):
    # #     v1, v2 = edge
    # #     size1 = mesh.sizing_values[v1.item()].item()
    # #     size2 = mesh.sizing_values[v2.item()].item()
    # #     size_formal = np.max([size1, size2])
    # #     related_points, related_dis = mesh.get_opposite_vertex(i)
    # #     related_size = mesh.sizing_values[related_points]      # + 
    # #     related_size = torch.min(related_size.squeeze(1) + related_dis * torch.log(torch.tensor(1.2)))
    # #     calculated_size = size[i]
    # #     true_size = torch.min(calculated_size, related_size)
    # #     rito = true_size / size_formal
    # #     if rito > 1.3:
    # #         result.append(1)
    # #     else:
    # #         result.append(0)

    # v1_list = mesh.edges[:, 0]
    # v2_list = mesh.edges[:, 1]
    # size1 = mesh.sizing_values[v1_list].squeeze()
    # size2 = mesh.sizing_values[v2_list].squeeze()
    # # size_formal = torch.mean(size1+size2)
    # size_formal = (size1 + size2) / 2
    # # for i in range(len(mesh.edges)):
    # #     related_points, related_dis, related_tris = mesh.get_opposite_vertex1(i)
    # #     # related_points, related_dis = mesh.get_opposite_vertex(i)
    # #     related_points_list.append(related_points)
    # #     related_dis_list.append(related_dis)
    # #     related_area_list.append(sum(mesh.area[related_tris]) / len(related_tris))

    # result = mesh.batch_process_edges()
    # related_points_list = result['related_points']
    # related_size = result['related_size']
    # related_area = result['related_area']
    # related_dis = result['related_dis']
        
    # # 转换为张量计算
    # # related_sizes = torch.stack([mesh.sizing_values[points] for points in related_points_list])
    # # related_dis = torch.stack(related_dis_list)
    # # related_size = torch.min(related_sizes.squeeze(-1) + related_dis * torch.log(torch.tensor(1.2)), dim=1).values
    # # related_area = torch.stack(related_area_list)

    # # 计算 true_size = min(calculated_size, related_size)
    # calculated_size = size  # 假设 mesh.size 存储了每条边的 calculated_size
    # # true_size = torch.minimum(calculated_size, related_size)
    # true_size = calculated_size

    # # for i in range(len(mesh.edges)):
    # #     print(i, calculated_size[i].item(), related_size[i].item(), size_formal[i])

    # # 计算 ratio = true_size / size_formal
    # ratio = true_size / size_formal

    # # 返回 ratio > 1.3 的布尔张量，并转换为 0/1 列表
    # # result = (ratio > 1.3).int().tolist()
    # result = torch.logical_and(
    #     torch.logical_and(torch.logical_and(true_size > size1, true_size > size2), ratio > 5),
    #     related_area > 1e-0
    # ).int().tolist()
        
    # # loop_time = time.time() - start_time
    # # print(f"{step} Edge processing loop time: {loop_time:.4f} s")

    # result = torch.tensor(result).unsqueeze(0)
    # f = torch.any(result == 1).item()
    # count = torch.count_nonzero(result == 1).item()
    # # print(count, "edges should be process")
    # # mesh = mesh.solve_banding2(result)

    # final_path1 = f"/home/zhuxunyang/coding/banding_detect/middle/final_refined_{step}.vtk"
    # final_path2 = f"/home/zhuxunyang/coding/banding_detect/middle/final_refined_sf_{step}.vtk"
    # save_path3 = f"/home/zhuxunyang/coding/banding_detect/middle/{step}_visual.vtk"
    # f = torch.any(result == 1).item()
    # if f:
    #     count = torch.count_nonzero(result == 1).item()
    #     print(count, "edges should be process")
    #     mesh.visual_edge(result.squeeze(0), save_path3)
    #     mesh = mesh.solve_banding2(result)
    #     mesh.writeVTK(final_path1)
    #     command1 = ['./remesh', '--input', final_path1, '--eps', '1e-4', '--envelope-dis', '1e-4', '--output', final_path1,'--split-num', '0', '--collapse-num', '0', '--feature-angle', '15']
    #     try:
    #         result1 = subprocess.run(command1, check=True, text=True, capture_output=True)
    #         print("remesh success")
    #     except subprocess.CalledProcessError as e:
    #         print("remesh fail")
    #         print(e.stderr)

    #     # this_mesh = CustomMesh.CustomMesh.from_vtk(final_path1)
    #     # find_nearest_triangles(this_mesh, mesh)
    #     # this_mesh.writeVTK(final_path1)

        # command = ['./test_sizefield_noIDmap', final_path1, final_path2, '1.2', '0', '0.1', geo_path]
        # try:
        #     result = subprocess.run(command, check=True, text=True, capture_output=True)
        # except subprocess.CalledProcessError as e:
        #     print(result.stdout)
        #     print("sizefield generate success")
        #     return None
    # else:
    #     mesh.writeVTK(final_path2)
    #     print("No edge banding")
    
    # step2_end_time = time.time() - step2_start_time
    # print("Step2 time:", step2_end_time)

    end_time = time.time()
    print(f"Total_Cost {end_time - begin_time:.2f}s")

def test_example_new(path, model_path, beta, theta):
    torch.set_float32_matmul_precision('high')
    total_time = defaultdict(float)
    total_start_time = time.time()
    print('Running Test')
    # opt = TestOptions().parse()
    # opt.serial_batches = True  # no shuffle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # val_dataset, norm_params = gather_graph_normal(path, opt, device, 1)
    val_dataset, _ = create_example1(path, device)
    # add_safe_globals([EdgeClassificationGNN, GCNConv])

    # model_gnn = EdgeClassificationGNN1(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64
    # ).to(device=device)
    # model_gnn = EdgeClassificationGNNWithEdgeAttention(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64
    # ).to(device=device)
    # model_gnn = EdgeClassificationGNNWithKANClassifier(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64
    # ).to(device=device)
    # model_gnn = EdgeClassificationGNNWithKANLayerAttention(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64
    # ).to(device=device)

    model_gnn = EdgeClassificationGNNWithKANLayerAttentionClassify2(
        node_feat_dim=1,
        edge_feat_dim=6,
        hidden_dim=64 
    ).to(device=device)

    # model_gnn = EdgeClassificationGNNWithKANLayerAttentionClassify1(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64 
    # ).to(device=device)
    model_gnn = torch.compile(model_gnn, dynamic=True, fullgraph=False)
    checkpoint = torch.load(
        model_path,
        weights_only=False
    )

    # model_gnn = EdgeRegressionGNNWithEdgeAttention(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64,
    # ).to(device=device)
    # checkpoint = torch.load(
    #     # "/home/zhuxunyang/coding/banding_detect/checkpoints/debug/GNN_new_Classify1_q1.02_new1.pth",
    #     "/home/zhuxunyang/coding/banding_detect/checkpoints/debug/GNNwithAttention_Regression.pth",
    #     weights_only=False
    # )

    model_gnn.load_state_dict(checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_batch = val_dataset.dataset[0]

    

    step = 0
    with torch.no_grad():
        start_time = time.time()
        # val_out = F.sigmoid(model_gnn(val_batch)).T
        val_out = F.sigmoid(model_gnn(val_batch.x, val_batch.edge_index, val_batch.edge_attr)).T
        classify_result = (val_out >= 0.5).float().to('cpu')
        print(torch.max(val_out), torch.min(val_out))
        geo_num = val_batch.geo

        target_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}_target.vtk"
        target_mesh = CustomMesh.CustomMesh.from_vtk(target_path)
        
        val_batch.mesh.get_all_info()
        count = torch.count_nonzero(classify_result == 1).item()
        print(count, "edges should be process")

        save_path1 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_bi.vtk"
        save_path2 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_bi_sf.vtk"
        save_path3 = f"/home/zhuxunyang/coding/banding_detect/middle1/visual.vtk"
        save_path4 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_bi_smooth.vtk"
        geo_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}.step"
        
        print("Predict-time:", time.time() - start_time, " s")

        val_batch.mesh.visual_edge(classify_result[0], save_path3)
        max_size = val_batch.mesh.get_max_size(1)

        # result_mesh = val_batch.mesh.solve_banding2(classify_result)    #ablation
        result_mesh, _,_ = val_batch.mesh.solve_banding6(classify_result, beta, theta)
        # result_mesh = val_batch.mesh.solve_banding_advanced_size(classify_result, beta, theta)

        result_mesh.get_all_info()
        result_mesh.writeVTK(save_path1)

        command1 = ['./remesh', '--input', save_path1, '--eps', '1e-2', '--envelope-dis', '5e-4', '--output', save_path4,'--split-num', '0', '--collapse-num', '0', '--feature-angle', '1']
        try:
            result1 = subprocess.run(command1, check=True, text=True, capture_output=True)
            print("remesh success")
            # print(result1.stdout)
        except subprocess.CalledProcessError as e:
            print("remesh fail")
            print(e.stderr)

        command = ['./test_sizefield', save_path4, save_path2, str(beta), str(max_size), str(theta), geo_path]
        # result = subprocess.run(command, check=True, text=True, capture_output=True)
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("size regenerated")
            # print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("size regenerated failed")
            print(e.stderr)
        result_mesh = CustomMesh.CustomMesh.from_vtk(save_path2, False)

        # result_mesh = CustomMesh.CustomMesh.from_vtk(save_path4, False)
        # result_mesh.recalculate_size(target_mesh)
        # result_mesh.writeVTK(save_path2)

        result_mesh.get_all_info()

        end_time = time.time()
        print(f"Total Cost {end_time - total_start_time:.2f}s")
        # num_target = target_mesh.num_mesh()
        num_before = val_batch.mesh.num_mesh()
        num_after = result_mesh.num_mesh()

    print(num_before, "/", num_after, "=", num_before / num_after)
    # print(num_before, "/", num_target, "=", num_before / num_target)
    
def test_example_new1(path, model_path, beta, theta):
    torch.set_float32_matmul_precision('high')
    total_time = defaultdict(float)
    total_start_time = time.time()
    print('Running Test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_dataset, _ = create_example1(path, device)
    # add_safe_globals([EdgeClassificationGNN, GCNConv])

    # model_gnn = EdgeClassificationGNN1(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64
    # ).to(device=device)
    # model_gnn = EdgeClassificationGNNWithEdgeAttention(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64
    # ).to(device=device)
    # model_gnn = EdgeClassificationGNNWithKANClassifier(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64
    # ).to(device=device)
    # model_gnn = EdgeClassificationGNNWithKANLayerAttention(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64
    # ).to(device=device)

    model_gnn = EdgeClassificationGNNWithKANLayerAttentionClassify2(
        node_feat_dim=1,
        edge_feat_dim=6,
        hidden_dim=64 
    ).to(device=device)

    # model_gnn = EdgeClassificationGNNWithKANLayerAttentionClassify4(
    #     node_feat_dim=1,
    #     edge_feat_dim=5,
    #     hidden_dim=64 
    # ).to(device=device)
    model_gnn = torch.compile(model_gnn, dynamic=True, fullgraph=True)
    checkpoint = torch.load(
        model_path,
        weights_only=False
    )

    # model_gnn = EdgeRegressionGNNWithEdgeAttention(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64,
    # ).to(device=device)
    # checkpoint = torch.load(
    #     # "/home/zhuxunyang/coding/banding_detect/checkpoints/debug/GNN_new_Classify1_q1.02_new1.pth",
    #     "/home/zhuxunyang/coding/banding_detect/checkpoints/debug/GNNwithAttention_Regression.pth",
    #     weights_only=False
    # )

    model_gnn.load_state_dict(checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_batch = val_dataset.dataset[0]

    

    step = 0
    with torch.no_grad():
        start_time = time.time()
        # val_out = F.sigmoid(model_gnn(val_batch)).T
        val_out = F.sigmoid(model_gnn(val_batch.x, val_batch.edge_index, val_batch.edge_attr)).T
        classify_result = (val_out >= 0.5).float().to('cpu')
        print(torch.max(val_out), torch.min(val_out))
        geo_num = val_batch.geo

        target_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}_target.vtk"
        # target_mesh = CustomMesh.CustomMesh.from_vtk(target_path)
        
        val_batch.mesh.get_all_info()
        count = torch.count_nonzero(classify_result == 1).item()
        print(count, "edges should be process")

        save_path1 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined.vtk"
        save_path2 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_sf.vtk"
        save_path3 = f"/home/zhuxunyang/coding/banding_detect/middle1/visual.vtk"
        save_path4 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_advance_sf.vtk"
        geo_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}.step"
        
        print("Predict-time:", time.time() - start_time, " s")

        val_batch.mesh.visual_edge(classify_result[0], save_path3)
        max_size = val_batch.mesh.get_max_size(1)

        # result_mesh, new_pos, edge_v_table = val_batch.mesh.solve_banding4(classify_result, beta, theta)
        # result_mesh_bi, new_pos, edge_v_table = val_batch.mesh.solve_banding6(classify_result, beta, theta)
        result_mesh, edge_v_table = val_batch.mesh.solve_banding7(classify_result)
        begin_mesh = val_batch.mesh

        result_mesh.get_all_info()
        result_mesh.writeVTK(save_path1)

        command = ['./test_sizefield', save_path1, save_path2, str(beta), str(max_size), str(theta), geo_path]
        # result = subprocess.run(command, check=True, text=True, capture_output=True)
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("size regenerated")
            # print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("size regenerated failed")
            print(e.stderr)

        refined_mesh = CustomMesh.CustomMesh.from_vtk(save_path2, False)
        refined_mesh.get_all_info()
        new_advanced_sizefield = begin_mesh.solve_banding_advanced_size(classify_result, refined_mesh, edge_v_table)


        # new_advanced_sizefield = optimize_mesh_sizing_unified(new_advanced_sizefield)
        # new_advanced_sizefield = smooth_advance_size(new_advanced_sizefield, beta)
        # new_advanced_sizefield.get_grad(beta)
        new_advanced_sizefield.writeVTK_with_advanced_info1(save_path4)
        new_advanced_sizefield.get_advance_info()

        # for i, key in enumerate(edge_v_table):
        #     print(refined_mesh.vertices[key[0]])
        #     print(refined_mesh.vertices[key[1]])
        #     advacne_id = edge_v_table[key][0]
        #     normal_v_len = len(new_advanced_sizefield.vertices)
        #     print(refined_mesh.vertices[advacne_id], new_advanced_sizefield.advanced_vertex[advacne_id - normal_v_len])
        #     print(" ")

        end_time = time.time()
        print(f"Total Cost {end_time - total_start_time:.2f}s")
        num_refined = refined_mesh.num_mesh()
        num_before = val_batch.mesh.num_mesh()
    #     num_after = result_mesh.num_mesh()

    print(model_path, num_before, "/", num_refined, "=", num_before / num_refined)
    
def test_example_new2(path, model_path, beta, theta):
    torch.set_float32_matmul_precision('high')
    total_time = defaultdict(float)
    total_start_time = time.time()
    print('Running Test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_dataset, _ = create_example1(path, device)
    
    model_gnn = EdgeRegressionGNNWithPureKAN(
        node_feat_dim=1,
        edge_feat_dim=6,
        hidden_dim=32,
    ).to(device=device)

    # model_gnn = torch.compile(model_gnn, dynamic=True, fullgraph=False)
    checkpoint = torch.load(
        model_path,
        weights_only=False
    )


    model_gnn.load_state_dict(checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_batch = val_dataset.dataset[0]

    

    step = 0
    with torch.no_grad():
        start_time = time.time()
        # val_out = F.sigmoid(model_gnn(val_batch)).T
        val_out = model_gnn(val_batch.x, val_batch.edge_index, val_batch.edge_attr)
        classify_result = (val_out >= 1.2).float().to('cpu').squeeze(1).unsqueeze(0)
        print(torch.max(val_out), torch.min(val_out))
        geo_num = val_batch.geo

        target_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}_target.vtk"
        target_mesh = CustomMesh.CustomMesh.from_vtk(target_path)
        
        val_batch.mesh.get_all_info()
        count = torch.count_nonzero(classify_result == 1).item()
        print(count, "edges should be process")

        save_path1 = f"/home/zhuxunyang/coding/banding_detect/middle/refined.vtk"
        save_path2 = f"/home/zhuxunyang/coding/banding_detect/middle/refined_sf.vtk"
        save_path3 = f"/home/zhuxunyang/coding/banding_detect/middle/visual.vtk"
        save_path4 = f"/home/zhuxunyang/coding/banding_detect/middle/refined_smooth.vtk"
        geo_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}.step"
        
        print("Predict-time:", time.time() - start_time, " s")

        val_batch.mesh.visual_edge(classify_result[0], save_path3)
        max_size = val_batch.mesh.get_max_size(1)

        # result_mesh = val_batch.mesh.solve_banding2(classify_result)    #ablation
        result_mesh, _,_ = val_batch.mesh.solve_banding6(classify_result, beta, theta)
        # result_mesh = val_batch.mesh.solve_banding_advanced_size(classify_result, beta, theta)

        result_mesh.get_all_info()
        result_mesh.writeVTK(save_path1)

        command1 = ['./remesh', '--input', save_path1, '--eps', '1e-2', '--envelope-dis', '5e-4', '--output', save_path4,'--split-num', '0', '--collapse-num', '0', '--feature-angle', '1']
        try:
            result1 = subprocess.run(command1, check=True, text=True, capture_output=True)
            print("remesh success")
            # print(result1.stdout)
        except subprocess.CalledProcessError as e:
            print("remesh fail")
            print(e.stderr)

        command = ['./test_sizefield', save_path4, save_path2, str(beta), str(max_size), str(theta), geo_path]
        # result = subprocess.run(command, check=True, text=True, capture_output=True)
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("size regenerated")
            # print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("size regenerated failed")
            print(e.stderr)
        result_mesh = CustomMesh.CustomMesh.from_vtk(save_path2, False)

        # result_mesh = CustomMesh.CustomMesh.from_vtk(save_path4, False)
        # result_mesh.recalculate_size(target_mesh)
        # result_mesh.writeVTK(save_path2)

        result_mesh.get_all_info()

        end_time = time.time()
        print(f"Total Cost {end_time - total_start_time:.2f}s")
        # num_target = target_mesh.num_mesh()
        num_before = val_batch.mesh.num_mesh()
        num_after = result_mesh.num_mesh()

    print(num_before, "/", num_after, "=", num_before / num_after)

def test_example_new3(path, model_path, beta, theta):
    torch.set_float32_matmul_precision('high')
    total_time = defaultdict(float)
    total_start_time = time.time()
    print('Running Test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_dataset, _ = create_example1(path, device)

    model_gnn = EdgeClassificationGNNWithKANLayerAttentionClassify2(
        node_feat_dim=1,
        edge_feat_dim=6,
        hidden_dim=64 
    ).to(device=device)

    model_gnn = torch.compile(model_gnn, dynamic=True, fullgraph=True)
    checkpoint = torch.load(
        model_path,
        weights_only=False
    )

    model_gnn.load_state_dict(checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_batch = val_dataset.dataset[0]

    step = 0
    with torch.no_grad():
        start_time = time.time()
        # val_out = F.sigmoid(model_gnn(val_batch)).T
        val_out = F.sigmoid(model_gnn(val_batch.x, val_batch.edge_index, val_batch.edge_attr)).T
        classify_result = (val_out >= 0.5).float().to('cpu')
        print(torch.max(val_out), torch.min(val_out))
        geo_num = val_batch.geo

        target_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}_target.vtk"
        # target_mesh = CustomMesh.CustomMesh.from_vtk(target_path)
        
        val_batch.mesh.get_all_info()
        count = torch.count_nonzero(classify_result == 1).item()
        print(count, "edges should be process")

        save_path1 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined.vtk"
        save_path2 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_sf_bi.vtk"
        save_path3 = f"/home/zhuxunyang/coding/banding_detect/middle1/visual.vtk"
        save_path4 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_advance_bi_sf.vtk"
        geo_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}.step"
        
        print("Predict-time:", time.time() - start_time, " s")

        val_batch.mesh.visual_edge(classify_result[0], save_path3)
        max_size = val_batch.mesh.get_max_size(1)

        # result_mesh, new_pos, edge_v_table = val_batch.mesh.solve_banding4(classify_result, beta, theta)
        result_mesh, new_pos, edge_v_table = val_batch.mesh.solve_banding6(classify_result, beta, theta)
        begin_mesh = val_batch.mesh

        result_mesh.get_all_info()
        result_mesh.writeVTK(save_path1)

        command = ['./test_sizefield', save_path1, save_path2, str(beta), str(max_size), str(theta), geo_path]
        # result = subprocess.run(command, check=True, text=True, capture_output=True)
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("size regenerated")
            # print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("size regenerated failed")
            print(e.stderr)

        refined_mesh = CustomMesh.CustomMesh.from_vtk(save_path2, False)
        refined_mesh.get_all_info()
        new_advanced_sizefield = begin_mesh.solve_banding_advanced_size(classify_result, refined_mesh, edge_v_table)


        # new_advanced_sizefield = optimize_mesh_sizing_unified(new_advanced_sizefield)
        # new_advanced_sizefield = smooth_advance_size(new_advanced_sizefield, beta)
        # new_advanced_sizefield.get_grad(beta)
        new_advanced_sizefield.writeVTK_with_advanced_info1(save_path4)
        new_advanced_sizefield.get_advance_info()

        # for i, key in enumerate(edge_v_table):
        #     print(refined_mesh.vertices[key[0]])
        #     print(refined_mesh.vertices[key[1]])
        #     advacne_id = edge_v_table[key][0]
        #     normal_v_len = len(new_advanced_sizefield.vertices)
        #     print(refined_mesh.vertices[advacne_id], new_advanced_sizefield.advanced_vertex[advacne_id - normal_v_len])
        #     print(" ")

        end_time = time.time()
        print(f"Total Cost {end_time - total_start_time:.2f}s")
        num_refined = refined_mesh.num_mesh()
        num_before = val_batch.mesh.num_mesh()
    #     num_after = result_mesh.num_mesh()

    print(model_path, num_before, "/", num_refined, "=", num_before / num_refined)

def test_example_new4(path, model_path, beta, theta):
    torch.set_float32_matmul_precision('high')
    total_time = defaultdict(float)
    total_start_time = time.time()
    print('Running Test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_dataset, _ = create_example1(path, device)

    # model_gnn = EdgeClassificationGNNWithKANLayerAttentionClassify2(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64 
    # ).to(device=device)

    # model_gnn = EdgeClassificationGNNWithKANLayerAttentionClassify2_AblationConv2(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64
    # ).to(device=device)

    model_gnn = EdgeClassificationGNNWithKANLayerAttentionClassify2_AblationAttention(
        node_feat_dim=1,
        edge_feat_dim=6,
        hidden_dim=64
    ).to(device=device)

    # model_gnn = EdgeClassificationGNNWithKANLayerAttentionClassify2_AblationClassifier1(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
    #     hidden_dim=64
    # ).to(device=device)

    model_gnn = torch.compile(model_gnn, dynamic=True, fullgraph=True)
    checkpoint = torch.load(
        model_path,
        weights_only=False
    )

    model_gnn.load_state_dict(checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_batch = val_dataset.dataset[0]

    step = 0
    with torch.no_grad():
        start_time = time.time()
        # val_out = F.sigmoid(model_gnn(val_batch)).T
        val_out = F.sigmoid(model_gnn(val_batch.x, val_batch.edge_index, val_batch.edge_attr)).T
        classify_result = (val_out >= 0.5).float().to('cpu')
        print(torch.max(val_out), torch.min(val_out))
        geo_num = val_batch.geo

        target_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}_target.vtk"
        # target_mesh = CustomMesh.CustomMesh.from_vtk(target_path)
        
        val_batch.mesh.get_all_info()
        count = torch.count_nonzero(classify_result == 1).item()
        print(count, "edges should be process")

        save_path1 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined.vtk"
        save_path2 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_bi.vtk"
        save_path3 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_sf.vtk"
        save_path4 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_sf_bi.vtk"
        save_path5 = f"/home/zhuxunyang/coding/banding_detect/middle1/visual.vtk"
        save_path6 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_advance_sf.vtk"
        save_path7 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_advance_bi_sf.vtk"
        geo_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}.step"
        
        print("Predict-time:", time.time() - start_time, " s")

        val_batch.mesh.visual_edge(classify_result[0], save_path5)
        max_size = val_batch.mesh.get_max_size(1)

        result_mesh_bi, new_pos, edge_v_table_bi = val_batch.mesh.solve_banding6(classify_result, beta, theta)
        result_mesh, edge_v_table = val_batch.mesh.solve_banding7(classify_result)
        begin_mesh = val_batch.mesh

        result_mesh.get_all_info()
        result_mesh.writeVTK(save_path1)

        result_mesh_bi.get_all_info()
        result_mesh_bi.writeVTK(save_path2)

        command = ['./test_sizefield', save_path1, save_path3, str(beta), str(max_size), str(theta), geo_path]
        # result = subprocess.run(command, check=True, text=True, capture_output=True)
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("size regenerated")
            # print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("size regenerated failed")
            print(e.stderr)

        command = ['./test_sizefield', save_path2, save_path4, str(beta), str(max_size), str(theta), geo_path]
        # result = subprocess.run(command, check=True, text=True, capture_output=True)
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("size regenerated")
            # print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("size regenerated failed")
            print(e.stderr)

        refined_mesh = CustomMesh.CustomMesh.from_vtk(save_path3, False)
        refined_mesh.get_all_info()
        new_advanced_sizefield = begin_mesh.solve_banding_advanced_size(classify_result, refined_mesh, edge_v_table)
        new_advanced_sizefield.writeVTK_with_advanced_info1(save_path6)
        new_advanced_sizefield.get_advance_info()

        refined_bi_mesh = CustomMesh.CustomMesh.from_vtk(save_path4, False)
        refined_bi_mesh.get_all_info()
        new_advanced_bi_sizefield = begin_mesh.solve_banding_advanced_size(classify_result, refined_bi_mesh, edge_v_table_bi)
        new_advanced_bi_sizefield.writeVTK_with_advanced_info1(save_path7)
        new_advanced_bi_sizefield.get_advance_info()

        end_time = time.time()
        print(f"Total Cost {end_time - total_start_time:.2f}s")
        num_refined = refined_mesh.num_mesh()
        num_refined_bi = refined_bi_mesh.num_mesh()
        num_before = val_batch.mesh.num_mesh()
    #     num_after = result_mesh.num_mesh()

    print(model_path, "sf:", num_before, "/", num_refined, "=", num_before / num_refined, " bisf:", num_before, "/", num_refined_bi, "=", num_before / num_refined_bi)

def test_example_new5(path, model_path, beta, theta):
    torch.set_float32_matmul_precision('high')
    total_time = defaultdict(float)
    total_start_time = time.time()
    print('Running Test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_dataset, _ = create_example1(path, device)

    model_gnn = EdgeClassificationGNNWithKANLayerAttentionClassify2(
        node_feat_dim=1,
        edge_feat_dim=7,
        hidden_dim=64 
    ).to(device=device)

    # model_gnn = EdgeClassificationGNNWithKANLayerAttentionClassify2_AblationConv2(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
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

    model_gnn = torch.compile(model_gnn, dynamic=True, fullgraph=True)
    checkpoint = torch.load(
        model_path,
        weights_only=False
    )

    model_gnn.load_state_dict(checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_batch = val_dataset.dataset[0]

    step = 0
    with torch.no_grad():
        start_time = time.time()
        # val_out = F.sigmoid(model_gnn(val_batch)).T
        val_out = F.sigmoid(model_gnn(val_batch.x, val_batch.edge_index, val_batch.edge_attr)).T
        classify_result = (val_out >= 0.5).float().to('cpu')
        print(torch.max(val_out), torch.min(val_out))
        geo_num = val_batch.geo

        target_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}_target.vtk"
        # target_mesh = CustomMesh.CustomMesh.from_vtk(target_path)
        
        val_batch.mesh.get_all_info()
        count = torch.count_nonzero(classify_result == 1).item()
        print(count, "edges should be process")

        save_path1 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined.vtk"
        save_path3 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_sf.vtk"
        save_path5 = f"/home/zhuxunyang/coding/banding_detect/middle1/visual.vtk"
        save_path6 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_advance_sf.vtk"
        geo_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}.step"
        
        print("Predict-time:", time.time() - start_time, " s")

        val_batch.mesh.visual_edge(classify_result[0], save_path5)
        max_size = val_batch.mesh.get_max_size(10)

        result_mesh, new_pos, edge_v_table = val_batch.mesh.solve_banding7(classify_result)
        begin_mesh = val_batch.mesh
        result_mesh.get_all_info()
        result_mesh.writeVTK(save_path1)

        command = ['./test_sizefield', save_path1, save_path3, str(beta), str(max_size), str(theta), geo_path]
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("size regenerated")
            # print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("size regenerated failed")
            print(e.stderr)

        refined_mesh = CustomMesh.CustomMesh.from_vtk(save_path3, False)
        refined_mesh.get_all_info()
        new_advanced_sizefield = begin_mesh.solve_banding_advanced_size1(classify_result, max_size, beta, edge_v_table, new_pos, refined_mesh)
        new_advanced_sizefield.writeVTK_with_advanced_info1(save_path6)
        new_advanced_sizefield.get_advance_info()

        end_time = time.time()
        print(f"Total Cost {end_time - total_start_time:.2f}s")
        num_refined = refined_mesh.num_mesh()
        num_before = val_batch.mesh.num_mesh()

    print(model_path, "sf:", num_before, "/", num_refined, "=", num_before / num_refined)

def test_example_new6(path, model_path, beta, theta):
    torch.set_float32_matmul_precision('high')
    total_time = defaultdict(float)
    total_start_time = time.time()
    print('Running Test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_dataset, _ = create_example1(path, device)

    model_gnn = EdgeClassificationGNNWithKANLayerAttentionClassify2(
        node_feat_dim=1,
        edge_feat_dim=6,
        hidden_dim=64 
    ).to(device=device)

    # model_gnn = EdgeClassificationGNNWithKANLayerAttentionClassify2_AblationConv2(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
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

    model_gnn = torch.compile(model_gnn, dynamic=True, fullgraph=True)
    checkpoint = torch.load(
        model_path,
        weights_only=False
    )

    model_gnn.load_state_dict(checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_batch = val_dataset.dataset[0]

    step = 0
    with torch.no_grad():
        start_time = time.time()
        # val_out = F.sigmoid(model_gnn(val_batch)).T
        val_out = F.sigmoid(model_gnn(val_batch.x, val_batch.edge_index, val_batch.edge_attr)).T
        classify_result = (val_out >= 0.5).float().to('cpu')
        print(torch.max(val_out), torch.min(val_out))
        geo_num = val_batch.geo

        target_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}_target.vtk"
        
        val_batch.mesh.get_all_info()
        count = torch.count_nonzero(classify_result == 1).item()
        print(count, "edges should be process")

        save_path1 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined.vtk"
        save_path2 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined1.vtk"
        save_path3 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_sf.vtk"
        save_path4 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_sf1.vtk"
        save_path5 = f"/home/zhuxunyang/coding/banding_detect/middle1/visual.vtk"
        save_path6 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_advance_sf.vtk"
        geo_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}.step"
        
        print("Predict-time:", time.time() - start_time, " s")

        val_batch.mesh.visual_edge(classify_result[0], save_path5)
        max_size = val_batch.mesh.get_max_size(10)
        
        time1 = time.time()
        result_mesh1, new_pos1, edge_v_table1 = val_batch.mesh.solve_banding8(classify_result)
        high_order_solve_time1 = time.time() - time1
        time2 = time.time()
        result_mesh2, new_pos2, edge_v_table2 = val_batch.mesh.solve_banding7(classify_result)
        solve_time1 = time.time() - time2
        
        begin_mesh = val_batch.mesh
        result_mesh1.get_all_info()
        result_mesh1.writeVTK(save_path1)
        result_mesh2.get_all_info()
        result_mesh2.writeVTK(save_path2)

        time3 = time.time()
        command = ['./test_sizefield', save_path1, save_path3, str(beta), str(max_size), str(theta), geo_path]
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("size regenerated")
            # print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("size regenerated failed")
            print(e.stderr)
        high_order_generate_time = time.time() - time3
        time4 = time.time()
        command = ['./test_sizefield', save_path2, save_path4, str(beta), str(max_size), str(theta), geo_path]
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("size regenerated")
            # print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("size regenerated failed")
            print(e.stderr)
        generate_time = time.time() - time4

        refined_mesh1 = CustomMesh.CustomMesh.from_vtk(save_path3, False)
        refined_mesh1.get_all_info()
        refined_mesh2 = CustomMesh.CustomMesh.from_vtk(save_path4, False)
        refined_mesh2.get_all_info()
        smooth_begin_time1 = time.time()
        smooth_mesh_sizing(refined_mesh1, len(begin_mesh.vertices), beta=beta, h_min= torch.min(refined_mesh1.sizing_values))
        smooth_time1 = time.time() - smooth_begin_time1
        print(f"Smooth cost {smooth_time1}s")
        smooth_begin_time2 = time.time()
        smooth_mesh_sizing(refined_mesh2, len(refined_mesh2.vertices), beta=beta, h_min= torch.min(refined_mesh2.sizing_values))
        smooth_time2 = time.time() - smooth_begin_time2
        print(f"Smooth cost {smooth_time2}s")
        time5 = time.time()
        new_advanced_sizefield = begin_mesh.solve_banding_advanced_size2(classify_result, max_size, beta, edge_v_table1, new_pos1, refined_mesh1)
        high_order_solve_time2 = time.time() - time5
        new_advanced_sizefield.writeVTK_with_advanced_info1(save_path6)
        new_advanced_sizefield.get_advance_info()

        refined_mesh2.writeVTK(save_path4)

        end_time = time.time()
        print(f"Total Cost {end_time - total_start_time:.2f}s")
        num_refined = refined_mesh1.num_mesh()
        num_before = val_batch.mesh.num_mesh()

    print(model_path, "sf:", num_before, "/", num_refined, "=", num_before / num_refined)
    print(high_order_solve_time1 + high_order_solve_time2, solve_time1)
    print(high_order_generate_time, generate_time)
    print(smooth_time1, smooth_time2)


def test_example_new7(path, model_path, beta, theta):
    torch.set_float32_matmul_precision('high')
    total_time = defaultdict(float)
    total_start_time = time.time()
    print('Running Test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_dataset, _ = create_example1(path, device)

    model_gnn = EdgeClassificationGNNWithKANLayerAttentionClassify2(
        node_feat_dim=1,
        edge_feat_dim=6,
        hidden_dim=64 
    ).to(device=device)

    # model_gnn = EdgeClassificationGNNWithKANLayerAttentionClassify2_AblationConv2(
    #     node_feat_dim=1,
    #     edge_feat_dim=6,
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

    model_gnn = torch.compile(model_gnn, dynamic=True, fullgraph=True)
    checkpoint = torch.load(
        model_path,
        weights_only=False
    )

    model_gnn.load_state_dict(checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_batch = val_dataset.dataset[0]

    step = 0
    with torch.no_grad():
        start_time = time.time()
        # val_out = F.sigmoid(model_gnn(val_batch)).T
        val_out = F.sigmoid(model_gnn(val_batch.x, val_batch.edge_index, val_batch.edge_attr)).T
        classify_result = (val_out >= 0.5).float().to('cpu')
        print(torch.max(val_out), torch.min(val_out))
        geo_num = val_batch.geo

        target_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}_target.vtk"
        
        val_batch.mesh.get_all_info()
        count = torch.count_nonzero(classify_result == 1).item()
        print(count, "edges should be process")

        save_path1 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined.vtk"
        save_path3 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_sf.vtk"
        save_path5 = f"/home/zhuxunyang/coding/banding_detect/middle1/visual.vtk"
        save_path6 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_advance_sf.vtk"
        geo_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}.step"
        
        print("Predict-time:", time.time() - start_time, " s")

        val_batch.mesh.visual_edge(classify_result[0], save_path5)
        max_size = val_batch.mesh.get_max_size(10)

        result_mesh, new_pos, edge_v_table = val_batch.mesh.solve_banding7(classify_result)
        begin_mesh = val_batch.mesh
        result_mesh.get_all_info()
        result_mesh.writeVTK(save_path1)

        command = ['./test_sizefield', save_path1, save_path3, str(beta), str(max_size), str(theta), geo_path]
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("size regenerated")
            # print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("size regenerated failed")
            print(e.stderr)

        refined_mesh = CustomMesh.CustomMesh.from_vtk(save_path3, False)
        refined_mesh.get_all_info()
        smooth_begin_time = time.time()
        smooth_mesh_sizing(refined_mesh, len(begin_mesh.vertices), beta=beta, h_min= torch.min(refined_mesh.sizing_values))
        print(f"Smooth cost {time.time() - smooth_begin_time}s")
        new_advanced_sizefield = begin_mesh.solve_banding_advanced_size2(classify_result, max_size, beta, edge_v_table, new_pos, refined_mesh)
        new_advanced_sizefield.writeVTK_with_advanced_info1(save_path6)
        new_advanced_sizefield.get_advance_info()

        end_time = time.time()
        print(f"Total Cost {end_time - total_start_time:.2f}s")
        num_refined = refined_mesh.num_mesh()
        num_before = val_batch.mesh.num_mesh()

    print(model_path, "sf:", num_before, "/", num_refined, "=", num_before / num_refined)

def test_compare_method(path):
    total_start_time = time.time()
    mesh = CustomMesh.CustomMesh.from_vtk(path)
    result = []
    geo_num = path.split('/')[-1].split('_')[0]
    geo_path = f"/home/zhuxunyang/coding/banding_detect/datasets/banding_dataset/test/{geo_num}/{geo_num}.step"

    Flag = True
    flag = 0
    step = 0
    while Flag:
        num_before = mesh.num_mesh()
        
        start_time = time.time()

        for edge in mesh.edges:
            edge_start_time = time.time()
            v1, v2 = edge
            size1 = mesh.sizing_values[v1.item()].item()
            size2 = mesh.sizing_values[v2.item()].item()
            size_formal = np.max([size1, size2])
            mid_point = (mesh.vertices[v1.item()] + mesh.vertices[v2.item()]) / 2
            process = subprocess.Popen(
                ['./test_querysize', str(mid_point[0].item()), str(mid_point[1].item()), str(mid_point[2].item()), path, geo_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
        
            if stderr:
                print("Error:", stderr.decode())
                return None
            
            output_lines = stdout.decode().splitlines()
            query_cost = str(output_lines[-2])
            size = float(output_lines[-1])  # 获取最后一行作为size值

            print(query_cost)

            if size / size_formal > 1.3:
                result.append(1)
            else:
                result.append(0)
            total_edge_time = time.time() - edge_start_time
            # print(total_edge_time, " s")
        
        loop_time = time.time() - start_time
        print(f"{step} Edge processing loop time: {loop_time:.4f} s")

        result = torch.tensor(result).unsqueeze(0)
        mesh = mesh.solve_banding(result)

        save_path1 = f"/home/zhuxunyang/coding/banding_detect/middle1/final_refined_{step}.vtk"
        save_path2 = f"/home/zhuxunyang/coding/banding_detect/middle1/final_refined_sf_{step}.vtk"
        mesh.writeVTK(save_path1)
        command1 = ['./remesh', '--input', save_path1, '--eps', '1e-4', '--envelope-dis', '1e-4', '--output', save_path1,'--split-num', '0', '--collapse-num', '0']
        try:
            result1 = subprocess.run(command1, check=True, text=True, capture_output=True)
            print("remesh success")
        except subprocess.CalledProcessError as e:
            print("remesh fail")
            print(e.stderr)

        # this_mesh = CustomMesh.CustomMesh.from_vtk(save_path1)
        # find_nearest_triangles(this_mesh, mesh)
        # this_mesh.writeVTK(save_path1)

        command = ['./test_sizefield_noIDmap', save_path1, save_path2, '1.2', '0', '0.1', geo_path]
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(result.stdout)
            return None
            
        mesh = CustomMesh.CustomMesh.from_vtk(save_path2)
        num_after = mesh.num_mesh()
        r = num_before / num_after

        if r < 1.05 and r >= 1:
            flag += 1
            print(step, r)
            step = step + 1
        elif r < 1:
            flag += 1
            print(f"The rito <= {r}, this refinement is not accept")
            continue
        else:
            flag = 0
            print(step, r)
            step = step + 1

        if flag >= 3:
            Flag = False
        
    total_time = time.time() - total_start_time
    print(total_time, " s")

def test_compare_method1(path):
    total_start_time = time.time()
    geo_num = path.split('/')[-1].split('_')[0]
    geo_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}.step"
    target_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}_target.vtk"
    target_mesh = CustomMesh.CustomMesh.from_vtk(target_path)

    save_path1 = f"/home/zhuxunyang/coding/banding_detect/middle1/general_refined.vtk"
    save_path2 = f"/home/zhuxunyang/coding/banding_detect/middle1/edge_visual.vtk"
    command2 = ['./remesh', '--input', path, '--eps', '1e-4', '--envelope-dis', '1e-4', '--output', save_path1, '--feature-angle', '1']
    # command2 = ['./remesh', '--input', path, '--eps', '1e-4', '--output', save_path1]
    try:
        result2 = subprocess.run(command2, check=True, text=True, capture_output=True)
        print("remesh success")
    except subprocess.CalledProcessError as e:
        print("remesh fail")
        print(e.stderr)

    mesh = CustomMesh.CustomMesh.from_vtk(save_path1)
    mesh.recalculate_size(target_mesh)
    mesh.get_all_info()

    result = []
    Flag = True
    flag = 0
    step = 0
    start_time = time.time()
    while Flag:
        num_before = sum(mesh.num_mesh())
        print("Querying size")
        size = []
        process = subprocess.Popen(
            ['./test_querysize', save_path1, geo_path, '0.1', '1.2'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        if stderr:
            print("Error:", stderr.decode())
            return None
        edge_num = len(mesh.edges)
        output_lines = stdout.decode().splitlines()
        for i in range(edge_num, 0, -1):
            # print(output_lines[-i])
            size.append(float(output_lines[-i]))
        size = torch.tensor(size)
        print("Query complete")
        assert len(size) == edge_num
        # for i, edge in enumerate(mesh.edges):
        #     v1, v2 = edge
        #     size1 = mesh.sizing_values[v1.item()].item()
        #     size2 = mesh.sizing_values[v2.item()].item()
        #     size_formal = np.max([size1, size2])
            # related_points, related_dis = mesh.get_oppolianggsite_vertex(i)
            # related_points, related_dis = mesh.get_opposite_vertex(i)
            # related_size = mesh.sizing_values[related_points]      # + 
            # related_size = torch.min(related_size.squeeze(1) + related_dis * torch.log(torch.tensor(1.2)))
            # calculated_size = size[i]
            # true_size = torch.min(calculated_size, related_size)
            # rito = true_size / size_formal
            # if rito > 1.3:
            #     result.append(1)
            # else:
            #     result.append(0)

            # 计算 related_size = min(related_sizes + related_dis * log(1.2))
        v1_list = mesh.edges[:, 0]
        v2_list = mesh.edges[:, 1]
        size1 = mesh.sizing_values[v1_list].squeeze()
        size2 = mesh.sizing_values[v2_list].squeeze()
        size_formal = torch.maximum(size1, size2)
        result = mesh.batch_process_edges()
        related_size = result['related_size']
        
        # 计算 true_size = min(calculated_size, related_size)
        calculated_size = size  # 假设 mesh.size 存储了每条边的 calculated_size
        true_size = torch.minimum(calculated_size, related_size)

        # 计算 ratio = true_size / size_formal
        ratio = true_size / size_formal

        # 返回 ratio > 1.3 的布尔张量，并转换为 0/1 列表
        result = (ratio > 1.3).int().tolist()

        
        loop_time = time.time() - start_time
        print(f"{step} Edge processing loop time: {loop_time:.4f} s")

        mesh.visual_edge(result, save_path2)
        result = torch.tensor(result).unsqueeze(0)
        f = torch.any(result == 1).item()
        count = torch.count_nonzero(result == 1).item()
        print(count, "edges should be process")
        mesh = mesh.solve_banding2(result)

        save_path1 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_{step}.vtk"
        save_path2 = f"/home/zhuxunyang/coding/banding_detect/middle1/refined_sf_{step}.vtk"
        if f:
            mesh.writeVTK(save_path1)
            command1 = ['./remesh', '--input', save_path1, '--eps', '1e-4', '--envelope-dis', '1e-4', '--output', save_path1,'--split-num', '0', '--collapse-num', '0', '--feature-angle', '1']
            try:
                result1 = subprocess.run(command1, check=True, text=True, capture_output=True)
                print("remesh success")
            except subprocess.CalledProcessError as e:
                print("remesh fail")
                print(e.stderr)
            command = ['./test_sizefield_noIDmap', save_path1, save_path2, '1.2', '0', '0.1', geo_path]
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(result.stdout)
                print("sizefield generate success")
                return None
                
            mesh = CustomMesh.CustomMesh.from_vtk(save_path2)
            num_after = sum(mesh.num_mesh())
            r = num_before / num_after

            Flag = False
        else:
            mesh.writeVTK(save_path2)
            print("No edge banding")
        
    total_time = time.time() - total_start_time
    print(total_time, " s")

def test_compare_method2(path, beta, theta):
    start_mesh = CustomMesh.CustomMesh.from_vtk(path)
    
    start_mesh.get_all_info()
    total_start_time = time.time()
    geo_num = path.split('/')[-1].split('_')[0]
    geo_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}.step"
    target_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}_target.vtk"
    target_mesh = CustomMesh.CustomMesh.from_vtk(target_path)

    # print(torch.min(start_mesh.sizing_values), torch.min(target_mesh.sizing_values))
    # print(torch.max(start_mesh.sizing_values), torch.max(target_mesh.sizing_values))


    save_path1 = f"/home/zhuxunyang/coding/banding_detect/middle1/general_refined.vtk"
    save_path2 = f"/home/zhuxunyang/coding/banding_detect/middle1/general_refined_sf.vtk"
    save_path3 = f"/home/zhuxunyang/coding/banding_detect/middle1/second_refine.vtk"
    save_path4 = f"/home/zhuxunyang/coding/banding_detect/middle1/second_refine_sf.vtk"

    command2 = ['./remesh', '--input', path, '--eps', '5e-4', '--envelope-dis', '1e-4', '--output', save_path1, '--feature-angle', '1']    #, '--feature-angle', '45'
    try:
        result2 = subprocess.run(command2, check=True, text=True, capture_output=True)
        print("remesh success")
        # print(result2.stdout)
    except subprocess.CalledProcessError as e:
        print("remesh fail")
        print(e.stderr)

    mesh = CustomMesh.CustomMesh.from_vtk(save_path1, False)
    mesh.recalculate_size2(target_mesh)
    # mesh.recalculate_size2(target_mesh)
    mesh.writeVTK(save_path2)
    mesh.get_all_info()
    max_size = mesh.get_max_size(10)

    result = []
    
    time1 = time.time()
    print("Querying size")
    size = []
    midpoints = []
    edge_num = len(mesh.edges)
    for i, edge in enumerate(mesh.edges):
        midpoints.append((mesh.vertices[edge[0]] + mesh.vertices[edge[1]]) / 2)

    midpoints = torch.stack(midpoints, dim=0)
    size = query_size(midpoints, target_mesh)
    print("Query complete")
    assert len(size) == edge_num

    # 计算 related_size = min(related_sizes + related_dis * log(1.2))
    v1_list = mesh.edges[:, 0]
    v2_list = mesh.edges[:, 1]
    size1 = mesh.sizing_values[v1_list].squeeze()
    size2 = mesh.sizing_values[v2_list].squeeze()
    size_formal = torch.maximum(size1, size2)

    # 计算 ratio = true_size / size_formal
    ratio = size / size_formal

    # 返回 ratio > 1.3 的布尔张量，并转换为 0/1 列表
    result = (ratio > 1.3).int().tolist()

    solve_time = time.time() - time1
    time2 = time.time()

    # mesh.visual_edge(result, save_path2)
    result = torch.tensor(result).unsqueeze(0)
    f = torch.any(result == 1).item()
    if not f:
        mesh.writeVTK(save_path4)
        print("No edge banding")
    else:
        count = torch.count_nonzero(result == 1).item()
        print(count, "edges should be process")
        # edge_indices = torch.tensor(torch.where(ratio > 1.3)[0].tolist()).unsqueeze(0)
        mesh = mesh.solve_banding2(result)
        mesh.writeVTK(save_path3)
        command = ['./test_sizefield', save_path3, save_path4, str(beta), str(max_size), str(theta), geo_path]
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("size regenerated")
            # print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("size regenerated failed")
            print(e.stderr)
        mesh = CustomMesh.CustomMesh.from_vtk(save_path4, False)
    
    generate_time = time.time() - time2
    refined_mesh = CustomMesh.CustomMesh.from_vtk(save_path4, False)
    time3 = time.time()
    smooth_mesh_sizing(refined_mesh, len(refined_mesh.vertices), beta=beta, h_min= torch.min(refined_mesh.sizing_values))
    smooth_time = time.time() - time3
    refined_mesh.writeVTK(save_path4)
    num_after = mesh.num_mesh()
    num_before = start_mesh.num_mesh()
    print(num_before, "/", num_after, "=", num_before / num_after)
    # num_target = target_mesh.num_mesh()
    # print(num_target, '/', num_after, "=", num_target / num_after)
        
    total_time = time.time() - total_start_time
    target_mesh
    print(total_time, "s")
    print(solve_time, generate_time, smooth_time)

def test_diangle_method(path, beta, theta):
    save_path1 = f"/home/zhuxunyang/coding/banding_detect/middle1/diangle_refined.vtk"
    save_path2 = f"/home/zhuxunyang/coding/banding_detect/middle1/diangle_refined_sf.vtk"

    total_start_time = time.time()
    start_mesh = CustomMesh.CustomMesh.from_vtk(path)
    start_mesh.get_all_info()
    edge_points = start_mesh._get_edge_points()
    dihedral_angle = start_mesh._compute_dihedral_angle(edge_points)
    result = (dihedral_angle >= 3.1415926).astype(int).tolist()
    classify_result = torch.tensor(result).unsqueeze(0)
    count = torch.count_nonzero(classify_result == 1).item()
    print(count, "edges should be process")
    result_mesh, _,_ = start_mesh.solve_banding6(classify_result, beta, theta)
    result_mesh.get_all_info()
    result_mesh.writeVTK(save_path1)
    
    geo_num = path.split('/')[-1].split('_')[0]
    geo_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}.step"
    # target_path = f"/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/target/{geo_num}_target.vtk"
    # target_mesh = CustomMesh.CustomMesh.from_vtk(target_path)

    command2 = ['./remesh', '--input', save_path1, '--eps', '1e-3', '--envelope-dis', '1e-4', '--output', save_path1,'--split-num', '0', '--collapse-num', '0', '--feature-angle', '1']    #, '--feature-angle', '45'
    try:
        result2 = subprocess.run(command2, check=True, text=True, capture_output=True)
        print("remesh success")
        # print(result2.stdout)
    except subprocess.CalledProcessError as e:
        print("remesh fail")
        print(e.stderr)
    max_size = result_mesh.get_max_size(1)

    command = ['./test_sizefield', save_path1, save_path2, str(beta), str(max_size), str(theta), geo_path]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("size regenerated")
        # print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("size regenerated failed")
        print(e.stderr)

    result_mesh = CustomMesh.CustomMesh.from_vtk(save_path2, False)
    result_mesh.get_all_info()
    smooth_mesh_sizing(result_mesh, len(result_mesh.vertices), beta=beta, h_min= torch.min(result_mesh.sizing_values))
    result_mesh.writeVTK(save_path2)
    
    num_after = result_mesh.num_mesh()
    num_before = start_mesh.num_mesh()
    print(num_before, "/", num_after, "=", num_before / num_after)
    # print(num_target, '/', num_after, "=", num_target / num_after)
        
    total_time = time.time() - total_start_time
    print(total_time, "s")


if __name__ == "__main__":
    # test_example1("/home/zhuxunyang/coding/banding_detect/datasets/test_case/238_-1_bkgm.vtk")
    # test_example2("/home/zhuxunyang/coding/banding_detect/datasets/test_case/271_-1_bkgm.vtk")
    # test_compare_method2("/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/test_case/277_bkgm.vtk", 5.0, 0.05)
    # test_diangle_method("/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/test_case/277_bkgm.vtk", 5.0, 0.05)
    # test_example_new("/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/test_case/215_bkgm.vtk", "/home/zhuxunyang/coding/banding_detect/checkpoints/debug/GNNKanAttentionClassify2_1115_Classify20_q1.05_new.pth", 20, 0.02)
    # test_example_new1("/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/test_case/237_bkgm.vtk", "/home/zhuxunyang/coding/banding_detect/checkpoints/debug/GNNKanAttentionClassify2_1117_Classify40_q1.05_new.pth", 10, 0.01)  #10.0
    # test_example_new3("/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/test_case/237_bkgm.vtk", "/home/zhuxunyang/coding/banding_detect/checkpoints/debug/GNNKanAttentionClassify2_1117_Classify40_q1.05_new.pth", 10, 0.01)
    # test_example_new4("/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/test_case/205_bkgm.vtk", "/home/zhuxunyang/coding/banding_detect/checkpoints/debug/GNNKanAttentionClassify2AblationAttention_1201_Classify40_q1.05_new.pth", 5.0, 0.05)  #10.0
    test_example_new6("/home/zhuxunyang/coding/banding_detect/datasets/circuit_test/test_case/263_bkgm.vtk", "/home/zhuxunyang/coding/banding_detect/checkpoints/debug/GNNKanAttentionClassify2_1117_Classify40_q1.02_new.pth", 1.2, 0.03)

    # mesh = CustomMesh.CustomMesh.from_vtk("/home/zhuxunyang/coding/banding_detect/datasets/model_expanded1109/data/1_0_bkgm.vtk")
    # edge_points = mesh._get_edge_points()
    # dia_angle = mesh._compute_dihedral_angle(edge_points)
    # print(dia_angle[384])
    



    