import os
import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad, pad_tensor, smooth_map
import numpy as np
from models.layers.CustomMesh import CustomMesh
import meshio
import re
from numpy.core.multiarray import _reconstruct
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
import pickle
import json
import hashlib
import random

def make_dataset(path):
        meshes = []
        target = []
        assert os.path.isdir(path), '%s is not a valid directory' % path

        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                if fname.endswith('_bkgm.vtk'):
                    path = os.path.join(root, fname)
                    meshes.append(path)
                elif fname.endswith('.pt'):
                    path = os.path.join(root, fname)
                    target.append(path)
                else:
                    continue


        def get_sort_key(filename):
            # 使用正则表达式匹配文件名开头的两个数字部分（格式如 "123_456_xxx" 或 "123_456.yyy"）
            filename = filename.split('/')[-1]
            # match = re.search(r'^(\d+)_(\d+)', filename)
            match = re.search(r'^([+-]?\d+)_([+-]?\d+)', filename)
            if match:
                # 提取开头的两个数字并转为元组 (int1, int2) 作为排序依据
                num1 = int(match.group(1))
                num2 = int(match.group(2))
                return (num1, num2)
            return (float('inf'), float('inf'))

        
        # print(len(meshes), len(target))

        meshes = sorted(meshes, key=get_sort_key)
        target = sorted(target, key=get_sort_key)

         
        return meshes, target

def load_or_compute_stats(all_node_features, all_edge_features, save_dir='norm_params'):
    os.makedirs(save_dir, exist_ok=True)
    node_file = os.path.join(save_dir, '/home/zhuxunyang/coding/banding_detect/para/node_stats.pkl')
    edge_file = os.path.join(save_dir, '/home/zhuxunyang/coding/banding_detect/para/edge_stats.pkl')

    # 尝试加载已有参数
    if os.path.exists(node_file) and os.path.exists(edge_file):
        with open(node_file, 'rb') as f:
            node_mean, node_std = pickle.load(f)
        with open(edge_file, 'rb') as f:
            edge_mean, edge_std = pickle.load(f)
        return node_mean, node_std, edge_mean, edge_std

    node_mean = np.mean(all_node_features, axis=0)
    node_std = np.std(all_node_features, axis=0) + 1e-8
    edge_mean = np.mean(all_edge_features, axis=1)
    edge_std = np.std(all_edge_features, axis=1) + 1e-8

    with open(node_file, 'wb') as f:
        pickle.dump((node_mean, node_std), f)
    with open(edge_file, 'wb') as f:
        pickle.dump((edge_mean, edge_std), f)
    return node_mean, node_std, edge_mean, edge_std

# def gather_graph_normal(path, opt, device, batch_size, norm_params=None):
#     vtk_path, pt_path = make_dataset(path)
#     # for i in range(len(pt_path)):
#     #     print(vtk_path[i], pt_path[i])
#     train_len = len(vtk_path)
#     graph_list = []
    
#     # 第一次遍历：仅当未提供归一化参数时收集特征
#     if norm_params is None:
#         all_node_features, all_edge_features = [], []
#         for i in range(train_len):
#             mesh = Mesh(file=vtk_path[i], opt=opt, hold_history=True, export_folder=opt.export_folder)
#             all_node_features.append(mesh.size_value)
#             all_edge_features.append(mesh.extract_features())
        

#         # 计算并保存全局归一化参数（或逐样本）
#         node_mean, node_std, edge_mean, edge_std = load_or_compute_stats(np.concatenate(all_node_features, axis=0), np.concatenate(all_edge_features, axis=1))
#     else:
#         node_mean, node_std, edge_mean, edge_std = norm_params  # 使用传入参数

#     # 第二次遍历：构建归一化的图数据
#     for i in range(train_len):
#         mesh = Mesh(file=vtk_path[i], opt=opt, hold_history=True, export_folder=opt.export_folder)
#         label = torch.load(pt_path[i])
#         target = smooth_map(label, 1.2).unsqueeze(0).to(device).float()
        
#         edge_len = len(mesh.edges)
#         target_edge_len = label.shape[0]
#         if edge_len != target_edge_len:
#             print(edge_len, target_edge_len)
#             print(f"{vtk_path[i]} and {pt_path[i]} error")
#             continue

#         edge_features = mesh.extract_features()

#         GEO = vtk_path[i].split('/')[-1].split('_')[0]
            
#         # 归一化（假设使用全局参数）
#         norm_node = (mesh.size_value - node_mean) / node_std
#         norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
        
        
    
#         data = Data(
#             x=torch.from_numpy(norm_node).float(),
#             edge_index=torch.from_numpy(mesh.edges).T.long(),
#             edge_attr=torch.from_numpy(norm_edge).T.float(),
#             pos=torch.from_numpy(mesh.vs),
#             y=target.T,
#             y1=(target.T > 0.5).float(),
#             # mesh=CustomMesh.from_vtk(vtk_path[i]),
#             geo = GEO
#         ).to(device)
#         graph_list.append(data)
    
#     return DataLoader(graph_list, batch_size=batch_size, shuffle=True), (node_mean, node_std, edge_mean, edge_std)


# def gather_graph_normal1(path, opt, device, batch_size, norm_params=None):
#     vtk_path, pt_path = make_dataset(path)
#     for i in range(len(pt_path)):
#         print(vtk_path[i], pt_path[i])
#     train_len = len(vtk_path)
#     graph_list = []

#     all_node_features, all_edge_features = [], []

#     # 第一次遍历：仅当未提供归一化参数时收集特征
#     if norm_params is None:
#         for i in range(train_len):
#             mesh = Mesh(file=vtk_path[i], opt=opt, hold_history=True, export_folder=opt.export_folder)
#             print(vtk_path[i], "loaded")
#             all_node_features.append(mesh.size_value)
#             all_edge_features.append(mesh.extract_features())
        

#         # 计算并保存全局归一化参数（或逐样本）
#         node_mean, node_std, edge_mean, edge_std = load_or_compute_stats(np.concatenate(all_node_features, axis=0), np.concatenate(all_edge_features, axis=1))
#     else:
#         node_mean, node_std, edge_mean, edge_std = norm_params  # 使用传入参数

#     # 第二次遍历：构建归一化的图数据
#     for i in range(train_len):
#         if len(all_node_features) != 0 and len(all_edge_features) != 0:
#             target = smooth_map(torch.load(pt_path[i]), 1.2).unsqueeze(0).to(device).float()
#             edge_features = all_edge_features[i]
#             GEO = vtk_path[i].split('/')[-1].split('_')[0]
#             norm_node = (all_node_features[i] - node_mean) / node_std
#             norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
#         else:
#             mesh = Mesh(file=vtk_path[i], opt=opt, hold_history=True, export_folder=opt.export_folder)
#             target = smooth_map(torch.load(pt_path[i]), 1.2).unsqueeze(0).to(device).float()
#             edge_features = mesh.extract_features()
#             GEO = vtk_path[i].split('/')[-1].split('_')[0]
            
#             # 归一化（假设使用全局参数）
#             norm_node = (mesh.size_value - node_mean) / node_std
#             norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)

        

        
#         data = Data(
#             x=torch.from_numpy(norm_node).float(),
#             edge_index=torch.from_numpy(mesh.edges).T.long(),
#             edge_attr=torch.from_numpy(norm_edge).T.float(),
#             pos=torch.from_numpy(mesh.vs),
#             y=target.T,
#             y1=(target.T > 0.5).float(),
#             # mesh=CustomMesh.from_vtk(vtk_path[i]),
#             geo = GEO
#         ).to(device)
#         graph_list.append(data)
    
#     return DataLoader(graph_list, batch_size=batch_size, shuffle=True), (node_mean, node_std, edge_mean, edge_std)

def get_graph_list_cache_filename(path, rito):
    params_str = f"{path}_{rito}"
    hash_obj = hashlib.md5(params_str.encode())
    return f"graph_list_cache_{hash_obj.hexdigest()}.pkl"

def shuffle_lists_together(list1, list2):
    """同时打乱两个列表，保持对应关系"""
    combined = list(zip(list1, list2))
    random.shuffle(combined)
    shuffled_list1, shuffled_list2 = zip(*combined)
    return list(shuffled_list1), list(shuffled_list2)

def gather_graph_normal2(path, device, batch_size, norm_params=None, rito=1.2, use_cache=True):
    graph_list_cache_file = get_graph_list_cache_filename(path, rito)
    norm_params_cache_file = graph_list_cache_file.replace('graph_list', 'norm_params')
    
    # 尝试从缓存加载图数据
    if use_cache and os.path.exists(graph_list_cache_file) and os.path.exists(norm_params_cache_file):
        print(f"Loading cached graph data from {graph_list_cache_file}")
        
        with open(norm_params_cache_file, 'r') as f:
            norm_params_data = json.load(f)
            node_mean, node_std, edge_mean, edge_std = (
                norm_params_data['node_mean'], 
                norm_params_data['node_std'],
                norm_params_data['edge_mean'], 
                norm_params_data['edge_std']
            )
            norm_params = (node_mean, node_std, edge_mean, edge_std)
        
        with open(graph_list_cache_file, 'rb') as f:
            graph_list = pickle.load(f)
        
        # 重新创建DataLoader（因为DataLoader包含随机状态）
        dataloader = DataLoader(graph_list, batch_size=batch_size, shuffle=True)
        return dataloader, norm_params


    vtk_path, pt_path = make_dataset(path)
    # vtk_path, pt_path = shuffle_lists_together(vtk_path, pt_path)
    train_len = len(vtk_path)
    graph_list = []
    
    # 第一次遍历：仅当未提供归一化参数时收集特征
    if norm_params is None:
        all_node_features, all_edge_features = [], []
        for i in range(train_len):
            mesh = CustomMesh.from_vtk(vtk_path[i])
            all_node_features.append(mesh.sizing_values)
            all_edge_features.append(mesh.compute_edge_features())
            # print(i)

        # 计算并保存全局归一化参数（或逐样本）
        node_mean, node_std, edge_mean, edge_std = load_or_compute_stats(np.concatenate(all_node_features, axis=0), np.concatenate(all_edge_features, axis=1))
    else:
        node_mean, node_std, edge_mean, edge_std = norm_params  # 使用传入参数

    # 第二次遍历：构建归一化的图数据
    for i in range(train_len):
        # mesh = Mesh(file=vtk_path[i], opt=opt, hold_history=True, export_folder=opt.export_folder)
        mesh = CustomMesh.from_vtk(vtk_path[i])
        label = torch.load(pt_path[i])
        # target = smooth_map(label, rito).unsqueeze(0).to(device).float()
        target = label.unsqueeze(0)
        # print(torch.min(target), torch.max(target))

        edge_len = len(mesh.edges)
        target_edge_len = label.shape[0]
        if edge_len != target_edge_len:
            print(edge_len, target_edge_len)
            print(f"{vtk_path[i]} and {pt_path[i]} error")
            continue
        # print(i)

        edge_features = mesh.compute_edge_features()

        GEO = vtk_path[i].split('/')[-1].split('_')[0]
            
        # 归一化（假设使用全局参数）
        norm_node = (mesh.sizing_values - node_mean) / node_std
        norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
    
        data = Data(
            x=norm_node.float(),
            edge_index=mesh.edges.T.long(),
            edge_attr=norm_edge.T.float(),
            pos=mesh.vertices,
            y=target.T.float(),
            y1=(target.T > rito).float(),
            # mesh=CustomMesh.from_vtk(vtk_path[i]),
            geo = GEO
        ).to(device)
        graph_list.append(data)
    
    # 缓存图数据和归一化参数
    if use_cache:
        # 缓存图数据列表
        with open(graph_list_cache_file, 'wb') as f:
            pickle.dump(graph_list, f)
        
        # 缓存归一化参数（使用JSON以便可读）
        norm_params_data = {
            'node_mean': node_mean.tolist() if hasattr(node_mean, 'tolist') else node_mean,
            'node_std': node_std.tolist() if hasattr(node_std, 'tolist') else node_std,
            'edge_mean': edge_mean.tolist() if hasattr(edge_mean, 'tolist') else edge_mean,
            'edge_std': edge_std.tolist() if hasattr(edge_std, 'tolist') else edge_std
        }
        with open(norm_params_cache_file, 'w') as f:
            json.dump(norm_params_data, f)
        
        # print(f"Graph data cached to {graph_list_cache_file}")
    
    dataloader = DataLoader(graph_list, batch_size=batch_size, shuffle=True)
    return dataloader, (node_mean, node_std, edge_mean, edge_std)

def gather_graph_normal3(path, opt, device, batch_size, norm_params=None, rito=1.2):
    vtk_path, pt_path = make_dataset(path)
    train_len = len(vtk_path)
    graph_list = []
    
    # 第一次遍历：仅当未提供归一化参数时收集特征
    if norm_params is None:
        all_node_features, all_edge_features = [], []
        for i in range(train_len):
            mesh = CustomMesh.from_vtk(vtk_path[i])
            all_node_features.append(mesh.sizing_values)
            all_edge_features.append(mesh.compute_edge_features())
            print(i)

        # 计算并保存全局归一化参数（或逐样本）
        node_mean, node_std, edge_mean, edge_std = load_or_compute_stats(np.concatenate(all_node_features, axis=0), np.concatenate(all_edge_features, axis=1))
    else:
        node_mean, node_std, edge_mean, edge_std = norm_params  # 使用传入参数

    # 第二次遍历：构建归一化的图数据
    for i in range(train_len):
        # mesh = Mesh(file=vtk_path[i], opt=opt, hold_history=True, export_folder=opt.export_folder)
        mesh = CustomMesh.from_vtk(vtk_path[i])
        label = torch.load(pt_path[i])
        edge_points = mesh._get_edge_points()

        edges_lengths = np.linalg.norm(mesh.vertices[edge_points[:, 0]] - mesh.vertices[edge_points[:, 1]], 
                                     ord=2, axis=1) + 1e-8
        original_target = torch.from_numpy(edges_lengths) * label
        target = original_target / torch.max(original_target)
        
        edge_len = len(mesh.edges)
        target_edge_len = label.shape[0]
        if edge_len != target_edge_len:
            print(edge_len, target_edge_len)
            print(f"{vtk_path[i]} and {pt_path[i]} error")
            continue
        # print(i)

        edge_features = mesh.compute_edge_features()

        GEO = vtk_path[i].split('/')[-1].split('_')[0]
            
        # 归一化（假设使用全局参数）
        norm_node = (mesh.sizing_values - node_mean) / node_std
        norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
    
        data = Data(
            x=norm_node.float(),
            edge_index=mesh.edges.T.long(),
            edge_attr=norm_edge.T.float(),
            pos=mesh.vertices,
            y=target.T.float(),
            y1=(target.T > 0.5).float(),
            # mesh=CustomMesh.from_vtk(vtk_path[i]),
            geo = GEO
        ).to(device)
        graph_list.append(data)
    
    return DataLoader(graph_list, batch_size=batch_size, shuffle=True), (node_mean, node_std, edge_mean, edge_std)
