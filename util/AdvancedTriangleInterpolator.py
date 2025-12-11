import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
# from models.layers.CustomMesh import CustomMesh

class AdvancedTriangleInterpolator:
    """
    支持边上1-2个高阶点的三角形插值器
    """
    
    def __init__(self, mesh):
        self.mesh = mesh
        self.edge_cache = {}  # 缓存边的高阶点信息
    
    def get_edge_points_info(self, edge: Tuple[torch.Tensor, torch.Tensor]) -> Dict:
        """
        获取边的高阶点信息
        
        返回:
            Dict with keys: 'count', 'positions', 'weights', 'params'
        """
        key = tuple(sorted((edge[0].item(), edge[1].item())))
        
        if key in self.edge_cache:
            return self.edge_cache[key]
        
        if key not in self.mesh.advanced_sizing_values:
            result = {'count': 0, 'positions': [], 'weights': [], 'params': []}
            self.edge_cache[key] = result
            return result
        
        point_indices = self.mesh.advanced_sizing_values[key]
        v1, v2 = edge[0].item(), edge[1].item()
        
        # 获取顶点权重
        w1 = self.mesh.sizing_values[int(v1)].item() if self.mesh.sizing_values is not None else 1.0
        w2 = self.mesh.sizing_values[int(v2)].item() if self.mesh.sizing_values is not None else 1.0
        
        positions = []
        weights = []
        params = []
        
        if len(point_indices) == 1:
            # 1个高阶点，默认在中间 (t=0.5)
            idx = point_indices[0]
            weight = self.mesh.sizing_values[idx].item() if self.mesh.sizing_values is not None else (w1 + w2) / 2
            positions.append((v1 + v2) / 2)  # 参数化位置
            weights.append(weight)
            params.append(0.5)
        else:
            # 2个高阶点，默认在1/3和2/3位置
            idx1, idx2 = point_indices
            weight1 = self.mesh.sizing_values[idx1].item() if self.mesh.sizing_values is not None else (2*w1 + w2) / 3
            weight2 = self.mesh.sizing_values[idx2].item() if self.mesh.sizing_values is not None else (w1 + 2*w2) / 3
            
            positions.extend([(2*v1 + v2) / 3, (v1 + 2*v2) / 3])
            weights.extend([weight1, weight2])
            params.extend([1/3, 2/3])
        
        result = {
            'count': len(point_indices),
            'positions': positions,
            'weights': weights,
            'params': params
        }
        self.edge_cache[key] = result
        return result
    
    def interpolate_at_point(self, point: torch.Tensor) -> float:
        """
        在网格中插值指定点的尺寸值
        
        参数:
            point: 查询点坐标 [x, y, z]
            
        返回:
            插值后的尺寸值
        """
        # 找到包含该点的三角形面
        target_face, barycentric = self._find_containing_face(point)
        if target_face is None:
            # 如果找不到包含的面，返回默认值或最近点的值
            return self._get_default_value(point)
        
        return self.interpolate_in_triangle(target_face, barycentric)
    

    # def batch_interpolate_sizing_values(self, points: torch.Tensor) -> torch.Tensor:
    #     """
    #     批量插值多个点的尺寸值
        
    #     参数:
    #         points: 查询点坐标张量 [N, 3]
            
    #     返回:
    #         尺寸值张量 [N]
    #     """
    #     results = []
    #     for i in range(len(points)):
    #         point = points[i]
    #         value = self.interpolate_sizing_value(point)
    #         results.append(value)
        
    #     return torch.tensor(results, dtype=points.dtype, device=points.device)


    def interpolate_in_triangle(self, face: Tuple[int, int, int], 
                               barycentric: Tuple[float, float, float]) -> float:
        """
        在三角形内进行插值，考虑边上的高阶点
        
        参数:
            face: 三角形三个顶点索引
            barycentric: 重心坐标 (λ1, λ2, λ3)
            
        返回:
            插值后的权重值
        """
        λ1, λ2, λ3 = barycentric
        
        # 获取三个顶点的权重
        v1, v2, v3 = face
        wA = self.mesh.sizing_values[v1].item() if self.mesh.sizing_values is not None else 1.0
        wB = self.mesh.sizing_values[v2].item() if self.mesh.sizing_values is not None else 1.0
        wC = self.mesh.sizing_values[v3].item() if self.mesh.sizing_values is not None else 1.0
        
        # 获取三条边的高阶点信息
        edges = [
            (self.mesh.vertices[v1], self.mesh.vertices[v2]),  # AB边
            (self.mesh.vertices[v2], self.mesh.vertices[v3]),  # BC边  
            (self.mesh.vertices[v3], self.mesh.vertices[v1])   # CA边
        ]
        
        edge_info = [self.get_edge_points_info(edge) for edge in edges]
        
        # 根据边上高阶点数量选择插值方法
        edge_counts = [info['count'] for info in edge_info]
        
        if all(count == 0 for count in edge_counts):
            # 没有高阶点，使用线性插值
            return self._linear_interpolation(λ1, λ2, λ3, wA, wB, wC)
        elif all(count == 1 for count in edge_counts):
            # 所有边都有1个高阶点，使用二次插值
            return self._quadratic_interpolation(λ1, λ2, λ3, wA, wB, wC, edge_info)
        elif all(count == 2 for count in edge_counts):
            # 所有边都有2个高阶点，使用三次插值
            return self._cubic_interpolation(λ1, λ2, λ3, wA, wB, wC, edge_info)
        else:
            # 混合情况，使用自适应插值
            return self._adaptive_interpolation(λ1, λ2, λ3, wA, wB, wC, edge_info)
    
    def _linear_interpolation(self, λ1, λ2, λ3, wA, wB, wC):
        """线性重心插值"""
        return λ1 * wA + λ2 * wB + λ3 * wC
    
    def _quadratic_interpolation(self, λ1, λ2, λ3, wA, wB, wC, edge_info):
        """二次插值 - 每条边1个高阶点"""
        # 获取边中点权重
        wD = edge_info[0]['weights'][0]  # AB边中点
        wE = edge_info[1]['weights'][0]  # BC边中点
        wF = edge_info[2]['weights'][0]  # CA边中点
        
        # 二次形函数
        NA = λ1 * (2 * λ1 - 1)
        NB = λ2 * (2 * λ2 - 1) 
        NC = λ3 * (2 * λ3 - 1)
        ND = 4 * λ1 * λ2
        NE = 4 * λ2 * λ3
        NF = 4 * λ3 * λ1
        
        return wA*NA + wB*NB + wC*NC + wD*ND + wE*NE + wF*NF
    
    def _cubic_interpolation(self, λ1, λ2, λ3, wA, wB, wC, edge_info):
        """三次插值 - 每条边2个高阶点"""
        # 获取边上两个点的权重
        wD1, wD2 = edge_info[0]['weights']  # AB边两个点
        wE1, wE2 = edge_info[1]['weights']  # BC边两个点
        wF1, wF2 = edge_info[2]['weights']  # CA边两个点
        
        # 三次形函数（简化版本）
        # 顶点形函数
        NA = λ1 * (3*λ1 - 1) * (3*λ1 - 2) / 2
        NB = λ2 * (3*λ2 - 1) * (3*λ2 - 2) / 2
        NC = λ3 * (3*λ3 - 1) * (3*λ3 - 2) / 2
        
        # 边点形函数
        ND1 = 9 * λ1 * λ2 * (3*λ1 - 1) / 2  # AB边第一个点
        ND2 = 9 * λ1 * λ2 * (3*λ2 - 1) / 2  # AB边第二个点
        NE1 = 9 * λ2 * λ3 * (3*λ2 - 1) / 2  # BC边第一个点
        NE2 = 9 * λ2 * λ3 * (3*λ3 - 1) / 2  # BC边第二个点
        NF1 = 9 * λ3 * λ1 * (3*λ3 - 1) / 2  # CA边第一个点
        NF2 = 9 * λ3 * λ1 * (3*λ1 - 1) / 2  # CA边第二个点
        
        return (wA*NA + wB*NB + wC*NC + 
                wD1*ND1 + wD2*ND2 + 
                wE1*NE1 + wE2*NE2 + 
                wF1*NF1 + wF2*NF2)
    
    def _adaptive_interpolation(self, λ1, λ2, λ3, wA, wB, wC, edge_info):
        """自适应插值 - 处理边上不同数量高阶点的混合情况"""
        # 基础线性部分
        base_weight = self._linear_interpolation(λ1, λ2, λ3, wA, wB, wC)
        
        # 边修正部分
        edge_corrections = 0
        total_edge_weight = 0
        
        # 三条边的参数
        edges_params = [
            (λ1, λ2, 0),      # AB边: λ1 + λ2 = 1, λ3 = 0
            (λ2, λ3, 1),      # BC边: λ2 + λ3 = 1, λ1 = 0  
            (λ3, λ1, 2)       # CA边: λ3 + λ1 = 1, λ2 = 0
        ]
        
        for α, β, edge_idx in edges_params:
            edge_weight = α + β
            if edge_weight > 1e-10:  # 避免除零
                info = edge_info[edge_idx]
                if info['count'] > 0:
                    # 计算该边的插值修正
                    correction = self._edge_correction(α/edge_weight, β/edge_weight, 
                                                     info, wA, wB, wC, edge_idx)
                    edge_corrections += correction * edge_weight
                    total_edge_weight += edge_weight
        
        if total_edge_weight > 1e-10:
            return base_weight + edge_corrections / total_edge_weight
        else:
            return base_weight
    
    def _edge_correction(self, t, u, edge_info, wA, wB, wC, edge_idx):
        """计算单边的插值修正"""
        v1_weight = [wA, wB, wC][edge_idx]
        v2_weight = [wB, wC, wA][(edge_idx + 1) % 3]
        
        # 线性插值基准
        linear_val = (1 - t) * v1_weight + t * v2_weight
        
        if edge_info['count'] == 1:
            # 1个高阶点的修正
            edge_weight = edge_info['weights'][0]
            edge_param = edge_info['params'][0]
            
            # 基于距离的修正
            dist_to_point = abs(t - edge_param)
            if dist_to_point < 0.5:
                influence = 1 - 2 * dist_to_point
                return (edge_weight - linear_val) * influence
            else:
                return 0
                
        else:  # count == 2
            # 2个高阶点的修正
            w1, w2 = edge_info['weights']
            p1, p2 = edge_info['params']
            
            # 使用有理函数插值
            if abs(t - p1) < 1e-10:
                return w1 - linear_val
            elif abs(t - p2) < 1e-10:
                return w2 - linear_val
            else:
                # 基于两个点的加权修正
                dist1 = abs(t - p1)
                dist2 = abs(t - p2)
                total_dist = dist1 + dist2
                if total_dist > 1e-10:
                    weight1 = dist2 / total_dist
                    weight2 = dist1 / total_dist
                    target_val = weight1 * w1 + weight2 * w2
                    return target_val - linear_val
                else:
                    return 0