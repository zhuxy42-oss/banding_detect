import torch
import numpy as np
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.linalg import spsolve, cg
from collections import defaultdict
import warnings

class UnifiedLaplaceOptimizer:
    """统一拉普拉斯优化器 - 同时处理普通点和高阶点"""
    
    def __init__(self, lambda_smooth=1.0, lambda_fidelity=0.3, 
                 max_iter=1000, tol=1e-4, use_cotangent_weights=True):
        self.lambda_smooth = lambda_smooth
        self.lambda_fidelity = lambda_fidelity
        self.max_iter = max_iter
        self.tol = tol
        self.use_cotangent_weights = use_cotangent_weights
    
    def build_unified_adjacency(self, mesh):
        """
        构建统一邻接关系：普通点 + 高阶点
        
        规则：
        1. 普通点与邻接边上的所有高阶点连接
        2. 高阶点与所在边的两个顶点和邻接面上的所有点（普通点+高阶点）连接
        """
        # 基本参数
        n_base = len(mesh.vertices)
        n_advanced = len(mesh.advanced_vertex) if mesh.advanced_vertex is not None else 0
        n_total = n_base + n_advanced
        
        adjacency_dict = defaultdict(list)
        
        # 步骤1: 构建普通点之间的基础邻接
        base_adjacency = self._build_base_adjacency(mesh)
        for i, neighbors in base_adjacency.items():
            adjacency_dict[i].extend(neighbors)
        
        if n_advanced > 0:
            # 步骤2: 普通点与邻接边上的高阶点连接
            self._connect_base_to_advanced(mesh, adjacency_dict, n_base)
            
            # 步骤3: 高阶点与所在边的顶点连接  
            self._connect_advanced_to_base(mesh, adjacency_dict, n_base)
            
            # 步骤4: 高阶点与邻接面上的所有点连接
            self._connect_advanced_to_advanced(mesh, adjacency_dict, n_base)
        
        return dict(adjacency_dict), n_total
    
    def _build_base_adjacency(self, mesh):
        """构建普通点之间的基础邻接关系"""
        edges_np = mesh.edges.cpu().numpy()
        adjacency = defaultdict(list)
        
        for edge in edges_np:
            i, j = edge[0], edge[1]
            adjacency[i].append(j)
            adjacency[j].append(i)
            
        return adjacency
    
    def _connect_base_to_advanced(self, mesh, adjacency_dict, n_base):
        """普通点连接到邻接边上的高阶点"""
        edge_map = mesh.advanced_vertex_edge_map
        advanced_vertices = mesh.advanced_vertex.cpu().numpy()
        
        # 为每条边建立高阶点列表
        # edge_to_advanced = defaultdict(list)
        # for adv_idx, edge_idx in enumerate(edge_map):
        #     # if edge_idx >= 0:
        #     edge_to_advanced[edge_idx].append(adv_idx)
        edge_to_advanced = mesh.advanced_vertex_edge_map
        
        # 连接普通点到对应边上的高阶点
        for edge_idx, advanced_indices in edge_to_advanced.items():
            # if edge_idx < len(edges_np):
            v1, v2 = edge_idx
            for adv_idx in advanced_indices:
                global_adv_idx = adv_idx
                adjacency_dict[v1].append(global_adv_idx)
                adjacency_dict[v2].append(global_adv_idx)
            # adjacency_dict[v1].append(advanced_indices)
            # adjacency_dict[v2].append(advanced_indices)

    def _connect_advanced_to_base(self, mesh, adjacency_dict, n_base):
        """高阶点连接到所在边的两个顶点"""
        edge_map = mesh.advanced_vertex_edge_map
        for index, key in enumerate(edge_map):
            v1, v2 = key
            global_adv_idx = edge_map[key]
            for idx in global_adv_idx:
                adjacency_dict[idx].extend([v1, v2])

        # for edge_idx, advanced_indices in edge_map.items():
        #     v1, v2 = edge_idx
        #     global_adv_idx = n_base + advanced_indices
        #     adjacency_dict[global_adv_idx].extend([v1, v2])
    
    def _connect_advanced_to_advanced(self, mesh, adjacency_dict, n_base):
        """高阶点连接到邻接面上的所有点（普通点+高阶点）"""
        edge_map = mesh.advanced_vertex_edge_map
        faces_np = mesh.faces.cpu().numpy()
        edge_to_faces = mesh._edge_face_map
        
        # 构建边索引到顶点对的映射
        edges_np = mesh.edges.cpu().numpy()
        edge_to_vertices = {}
        for edge_idx, edge in enumerate(edges_np):
            edge_to_vertices[edge_idx] = tuple(sorted(edge))
        
        # 对于每个高阶点，找到其所在边的邻接面，然后连接面上的所有点
        for i, key in enumerate(edge_map):
            if key in edge_to_faces:
                face_indices = edge_to_faces[key]
                # global_adv_idx = n_base + i
                global_adv_idx = edge_map[key]
                for global_v in global_adv_idx:
                    connected_points = set()
                    for face_idx in face_indices:
                        face = faces_np[face_idx]
                        # 添加面上的普通点
                        connected_points.update(face)
                        # 添加面上的高阶点（通过面的边查找）
                        for v1, v2 in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
                            edge_key = tuple(sorted([v1, v2]))
                            if edge_key in edge_map:
                                for v in edge_map[edge_key]:
                                    connected_points.add(v)
                    # print(global_v, connected_points)
                    adjacency_dict[global_v].extend(list(connected_points))
    
    def build_unified_laplacian(self, adjacency_dict, n_total, vertices=None):
        """构建统一的拉普拉斯矩阵"""
        I, J, V = [], [], []
        
        if self.use_cotangent_weights and vertices is not None:
            # 使用cotangent权重（需要几何信息）
            return self._build_cotangent_laplacian_unified(adjacency_dict, n_total, vertices)
        else:
            # 使用组合拉普拉斯（均匀权重）
            return self._build_combinatorial_laplacian_unified(adjacency_dict, n_total)
    
    def _build_combinatorial_laplacian_unified(self, adjacency_dict, n_total):
        """构建组合拉普拉斯矩阵"""
        I, J, V = [], [], []
        
        for i, neighbors in adjacency_dict.items():
            degree = len(neighbors)
            # 对角线元素（度）
            I.append(i)
            J.append(i)
            V.append(degree)
            
            # 非对角线元素
            for j in neighbors:
                I.append(i)
                J.append(j)
                V.append(-1.0)
        
        return coo_matrix((V, (I, J)), shape=(n_total, n_total))
    
    def _build_cotangent_laplacian_unified(self, adjacency_dict, n_total, vertices):
        """构建cotangent权重的统一拉普拉斯矩阵"""
        # 注意：这里简化实现，实际应用中需要更精细的cotangent权重计算
        # 对于包含高阶点的系统，cotangent权重的计算会复杂很多
        I, J, V = [], [], []
        weight_sums = np.zeros(n_total)
        
        # 这里使用距离倒数作为权重近似
        vertices_np = vertices.cpu().numpy()
        
        for i, neighbors in adjacency_dict.items():
            pos_i = vertices_np[i]
            total_weight = 0.0
            
            for j in neighbors:
                pos_j = vertices_np[j]
                distance = np.linalg.norm(pos_j - pos_i)
                weight = 1.0 / (distance + 1e-8)
                
                I.append(i)
                J.append(j)
                V.append(-weight)
                total_weight += weight
            
            I.append(i)
            J.append(i)
            V.append(total_weight)
        
        return coo_matrix((V, (I, J)), shape=(n_total, n_total))
    
    def create_unified_sizing_vector(self, mesh):
        """创建统一的尺寸向量"""
        n_base = len(mesh.vertices)
        n_advanced = len(mesh.advanced_vertex) if mesh.advanced_vertex is not None else 0
        n_total = n_base + n_advanced
        
        unified_sizing = np.zeros(n_total)
        
        # 填充普通点尺寸
        if mesh.sizing_values is not None:
            base_sizing = mesh.sizing_values.cpu().numpy()
            unified_sizing[:n_base] = base_sizing[:, 0]
        else:
            # 如果没有基础尺寸，使用默认值
            unified_sizing[:n_base] = 1.0
        
        # 填充高阶点尺寸
        if n_advanced > 0 and mesh.advanced_size is not None:
            advanced_sizing = mesh.advanced_size.cpu().numpy()
            unified_sizing[n_base:n_total] = advanced_sizing[:, 0]
        elif n_advanced > 0:
            # 如果没有高阶尺寸，从相邻普通点插值
            for adv_idx in range(n_advanced):
                global_idx = n_base + adv_idx
                # 找到相邻的普通点
                edge_idx = mesh.advanced_vertex_edge_map[adv_idx].item()
                if edge_idx >= 0:
                    v1, v2 = mesh.edges[edge_idx].cpu().numpy()
                    # 简单平均
                    unified_sizing[global_idx] = 0.5 * (unified_sizing[v1] + unified_sizing[v2])
        
        return torch.tensor(unified_sizing, dtype=torch.float32)
    
    def split_unified_sizing_vector(self, unified_sizing, mesh):
        """将统一尺寸向量拆分为普通点和高阶点尺寸"""
        n_base = len(mesh.vertices)
        n_advanced = len(mesh.advanced_vertex) if mesh.advanced_vertex is not None else 0
        
        unified_np = unified_sizing.cpu().numpy()
        
        base_sizing = torch.tensor(unified_np[:n_base], dtype=torch.float32)
        
        if n_advanced > 0:
            advanced_sizing = torch.tensor(unified_np[n_base:n_base+n_advanced], dtype=torch.float32)
        else:
            advanced_sizing = None
            
        return base_sizing, advanced_sizing
    
    def optimize_unified_sizing(self, mesh, feature_mask=None):
        """
        统一优化普通点和高阶点尺寸场
        
        参数:
        mesh: CustomMesh对象
        feature_mask: 特征点掩码（只对普通点有效）
        
        返回:
        优化后的普通点尺寸和高阶点尺寸
        """
        print("Building unified adjacency...")
        adjacency_dict, n_total = self.build_unified_adjacency(mesh)
        
        print("Creating unified sizing vector...")
        unified_sizing = self.create_unified_sizing_vector(mesh)
        
        print("Building unified Laplacian...")
        # 合并所有点的坐标
        if mesh.advanced_vertex is not None:
            all_vertices = torch.cat([mesh.vertices, mesh.advanced_vertex], dim=0)
        else:
            all_vertices = mesh.vertices
        
        L = self.build_unified_laplacian(adjacency_dict, n_total, all_vertices)
        
        print("Solving optimization problem...")
        optimized_sizing = self._solve_laplace_system(
            unified_sizing.cpu().numpy(), L, n_total, feature_mask, mesh
        )
        
        print("Splitting results...")
        base_sizing, advanced_sizing = self.split_unified_sizing_vector(
            torch.tensor(optimized_sizing, dtype=torch.float32), mesh
        )
        
        return base_sizing, advanced_sizing
    
    def _solve_laplace_system(self, sizing_np, L, n_total, feature_mask, mesh):
        """求解拉普拉斯系统"""
        # 构建优化系统: (λ_smooth * L^T L + λ_fidelity * I) x = λ_fidelity * x0
        A = self.lambda_smooth * L.T @ L + self.lambda_fidelity * eye(n_total)
        b = self.lambda_fidelity * sizing_np.copy()
        
        # 处理特征点约束（只对普通点）
        if feature_mask is not None:
            feature_indices = torch.where(feature_mask)[0].cpu().numpy()
            for idx in feature_indices:
                if idx < n_total:  # 确保是普通点
                    A[idx, :] = 0
                    A[idx, idx] = 1.0
                    b[idx] = sizing_np[idx]
        
        # 求解
        try:
            optimized_sizing = spsolve(A, b)
        except:
            warnings.warn("Direct solve failed, using iterative method")
            optimized_sizing = self._iterative_laplace_smooth(
                sizing_np, L, n_total, feature_mask
            )
        
        return optimized_sizing
    
    def _iterative_laplace_smooth(self, sizing_np, L, n_total, feature_mask, num_iterations=50):
        """迭代拉普拉斯光滑化"""
        current_sizing = sizing_np.copy()
        
        for iteration in range(num_iterations):
            # 拉普拉斯更新: x_new = x - α * L x
            laplacian_update = L.dot(current_sizing)
            alpha = 0.5 * (1.0 - iteration / num_iterations)
            new_sizing = current_sizing - alpha * laplacian_update
            
            # 保持特征点不变
            if feature_mask is not None:
                feature_indices = torch.where(feature_mask)[0].cpu().numpy()
                for idx in feature_indices:
                    if idx < n_total:
                        new_sizing[idx] = sizing_np[idx]
            
            # 检查收敛
            diff_norm = np.linalg.norm(new_sizing - current_sizing)
            if diff_norm < self.tol:
                print(f"Converged after {iteration} iterations")
                break
                
            current_sizing = new_sizing
        
        return current_sizing

    def enforce_unified_constraints(self, base_sizing, advanced_sizing, mesh, 
                                  min_ratio=0.1, max_ratio=10.0, max_gradient=2.0):
        """对统一尺寸场施加物理约束"""
        # 合并尺寸场进行约束处理
        if advanced_sizing is not None:
            unified = torch.cat([base_sizing, advanced_sizing])
        else:
            unified = base_sizing
            
        if mesh.advanced_vertex is not None:
            all_vertices = torch.cat([mesh.vertices, mesh.advanced_vertex])
        else:
            all_vertices = mesh.vertices
        
        # 计算特征长度
        bbox_min = torch.min(all_vertices, dim=0)[0]
        bbox_max = torch.max(all_vertices, dim=0)[0]
        bbox_diag = torch.norm(bbox_max - bbox_min)
        char_length = bbox_diag * 0.01
        
        # 应用尺寸界限
        min_size = char_length * min_ratio
        max_size = char_length * max_ratio
        constrained_unified = torch.clamp(unified, min_size, max_size)
        
        # 拆分结果
        n_base = len(mesh.vertices)
        constrained_base = constrained_unified[:n_base]
        
        if advanced_sizing is not None:
            constrained_advanced = constrained_unified[n_base:]
        else:
            constrained_advanced = None
            
        return constrained_base, constrained_advanced
    

class Gradinet(UnifiedLaplaceOptimizer):
    def __init__(self, mesh, beta):
        self.vertex_table = None
        self.mesh = mesh
        self.beta = beta

    def calculate_gradinet(self):
        n_base = len(self.mesh.vertices)
        n_advanced = len(self.mesh.advanced_vertex) if self.mesh.advanced_vertex is not None else 0
        n_total = n_base + n_advanced
        
        adjacency_dict = defaultdict(list)
        
        # 步骤1: 构建普通点之间的基础邻接
        base_adjacency = self._build_base_adjacency(self.mesh)
        for i, neighbors in base_adjacency.items():
            adjacency_dict[i].extend(neighbors)
        
        if n_advanced > 0:
            # 步骤2: 普通点与邻接边上的高阶点连接
            self._connect_base_to_advanced(self.mesh, adjacency_dict, n_base)
            
            # 步骤3: 高阶点与所在边的顶点连接  
            self._connect_advanced_to_base(self.mesh, adjacency_dict, n_base)
            
            # 步骤4: 高阶点与邻接面上的所有点连接
            self._connect_advanced_to_advanced(self.mesh, adjacency_dict, n_base)
        yuzhi = np.log(self.beta)
        for i, key in enumerate(adjacency_dict):
            v_list = adjacency_dict[key]
            if key >= n_base:
                size1 = self.mesh.advanced_size[key - n_base]
                p1 = self.mesh.advanced_vertex[key - n_base]
            else:
                size1 = self.mesh.sizing_values[key]
                p1 = self.mesh.vertices[key]
            for v in v_list:
                if v >= n_base:
                    size2 = self.mesh.advanced_size[v - n_base]
                    length = torch.norm(p1 - self.mesh.advanced_vertex[v - n_base])
                else:
                    size2 = self.mesh.sizing_values[v]
                    length = torch.norm(p1 - self.mesh.vertices[v])
                grad = torch.abs(size1 - size2) / length
                if grad > yuzhi:
                    print(v, key, size1.item(), size2.item(), length.item(), grad.item())


            


class OneWayLaplaceOptimizer(UnifiedLaplaceOptimizer):
    """只缩小尺寸的拉普拉斯优化器"""
    
    def optimize_unidirectional(self, mesh, feature_mask=None, max_reduction_ratio=0.3):
        """
        单向优化：只减小尺寸
        
        参数:
        mesh: 网格对象
        feature_mask: 特征点掩码
        max_reduction_ratio: 最大缩小比例 (0.3 = 最多缩小30%)
        """
        # 构建统一邻接关系
        adjacency_dict, n_total = self.build_unified_adjacency(mesh)
        unified_sizing = self.create_unified_sizing_vector(mesh)
        
        # 构建拉普拉斯矩阵
        if mesh.advanced_vertex is not None:
            all_vertices = torch.cat([mesh.vertices, mesh.advanced_vertex], dim=0)
        else:
            all_vertices = mesh.vertices
        
        L = self.build_unified_laplacian(adjacency_dict, n_total, all_vertices)
        
        # 单向优化
        optimized_sizing = self._solve_unidirectional_system(
            unified_sizing.cpu().numpy(), L, n_total, feature_mask, mesh, max_reduction_ratio
        )
        
        # 拆分结果
        base_sizing, advanced_sizing = self.split_unified_sizing_vector(
            torch.tensor(optimized_sizing, dtype=torch.float32), mesh
        )
        
        return base_sizing, advanced_sizing
    
    def build_unified_adjacency(self, mesh):
        """
        构建统一邻接关系：普通点 + 高阶点
        
        规则：
        1. 普通点与邻接边上的所有高阶点连接
        2. 高阶点与所在边的两个顶点和邻接面上的所有点（普通点+高阶点）连接
        """
        # 基本参数
        n_base = len(mesh.vertices)
        n_advanced = len(mesh.advanced_vertex) if mesh.advanced_vertex is not None else 0
        n_total = n_base + n_advanced
        
        adjacency_dict = defaultdict(list)
        
        # 步骤1: 构建普通点之间的基础邻接
        base_adjacency = self._build_base_adjacency(mesh)
        for i, neighbors in base_adjacency.items():
            adjacency_dict[i].extend(neighbors)
        
        if n_advanced > 0:
            # 步骤2: 普通点与邻接边上的高阶点连接
            self._connect_base_to_advanced(mesh, adjacency_dict, n_base)
            
            # 步骤3: 高阶点与所在边的顶点连接  
            self._connect_advanced_to_base(mesh, adjacency_dict, n_base)
            
            # 步骤4: 高阶点与邻接面上的所有点连接
            self._connect_advanced_to_advanced(mesh, adjacency_dict, n_base)
        
        return dict(adjacency_dict), n_total

    def create_unified_sizing_vector(self, mesh):
        """创建统一的尺寸向量"""
        n_base = len(mesh.vertices)
        n_advanced = len(mesh.advanced_vertex) if mesh.advanced_vertex is not None else 0
        n_total = n_base + n_advanced
        
        unified_sizing = np.zeros(n_total)
        
        # 填充普通点尺寸
        if mesh.sizing_values is not None:
            base_sizing = mesh.sizing_values.cpu().numpy()
            unified_sizing[:n_base] = base_sizing[:, 0]
        else:
            # 如果没有基础尺寸，使用默认值
            unified_sizing[:n_base] = 1.0
        
        # 填充高阶点尺寸
        if n_advanced > 0 and mesh.advanced_size is not None:
            advanced_sizing = mesh.advanced_size.cpu().numpy()
            unified_sizing[n_base:n_total] = advanced_sizing[:, 0]
        elif n_advanced > 0:
            # 如果没有高阶尺寸，从相邻普通点插值
            for adv_idx in range(n_advanced):
                global_idx = n_base + adv_idx
                # 找到相邻的普通点
                edge_idx = mesh.advanced_vertex_edge_map[adv_idx].item()
                if edge_idx >= 0:
                    v1, v2 = mesh.edges[edge_idx].cpu().numpy()
                    # 简单平均
                    unified_sizing[global_idx] = 0.5 * (unified_sizing[v1] + unified_sizing[v2])
        
        return torch.tensor(unified_sizing, dtype=torch.float32)

    def _solve_unidirectional_system(self, sizing_np, L, n_total, feature_mask, mesh, max_reduction_ratio):
        """求解单向优化系统"""
        from scipy.optimize import minimize
        import numpy as np
        
        original_sizing = sizing_np.copy()
        
        # 定义优化目标函数
        def objective(x):
            # 光滑项: 最小化拉普拉斯能量
            smooth_term = self.lambda_smooth * (x.T @ L.T @ L @ x)
            
            # 保真项: 最小化与原始尺寸的偏差（但只惩罚增大的情况）
            increase_mask = x > original_sizing
            fidelity_term = self.lambda_fidelity * np.sum(
                (x[increase_mask] - original_sizing[increase_mask])**2
            )
            
            return smooth_term + fidelity_term
        
        # 定义约束：尺寸只能减小，不能增大
        constraints = []
        for i in range(n_total):
            if feature_mask is not None and i < len(feature_mask) and feature_mask[i]:
                # 特征点：尺寸完全固定
                constraints.append({'type': 'eq', 'fun': lambda x, i=i: x[i] - original_sizing[i]})
            else:
                # 非特征点：尺寸只能减小
                min_size = original_sizing[i] * (1 - max_reduction_ratio)
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i, min_size=min_size: x[i] - min_size})
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: original_sizing[i] - x[i]})
        
        # 初始猜测（稍微缩小一点）
        x0 = original_sizing * (1 - 0.1 * max_reduction_ratio)
        
        # 求解约束优化问题
        result = minimize(objective, x0, method='SLSQP', constraints=constraints, 
                         options={'maxiter': self.max_iter})
        
        if result.success:
            return result.x
        else:
            print(f"优化失败: {result.message}")
            return original_sizing  # 返回原始尺寸
        


# class ShrinkingLaplaceOptimizer(UnifiedLaplaceOptimizer):
#     """基于邻接点尺寸和梯度约束的高阶点尺寸平滑优化器"""
    
#     def optimize_advanced_gradient_control(self, mesh, beta=1.5, feature_mask=None, num_iterations=50):
#         """
#         只平滑高阶点尺寸并控制梯度的优化
        
#         参数:
#         mesh: 网格对象
#         beta: 梯度控制参数，梯度控制在 log(beta) 范围内
#         feature_mask: 特征点掩码
#         num_iterations: 迭代次数
#         """
#         # 构建统一邻接关系
#         adjacency_dict, n_total = self.build_unified_adjacency(mesh)
#         unified_sizing = self.create_unified_sizing_vector(mesh)
#         current_sizing = unified_sizing.cpu().numpy()
#         original_sizing = current_sizing.copy()
        
#         # 获取普通点和高阶点的索引范围
#         n_base = len(mesh.vertices)
#         n_advanced = len(mesh.advanced_vertex) if mesh.advanced_vertex is not None else 0
#         base_indices = list(range(n_base))
#         advanced_indices = list(range(n_base, n_base + n_advanced))
        
#         # 获取顶点位置用于距离计算
#         if mesh.advanced_vertex is not None:
#             all_vertices = torch.cat([mesh.vertices, mesh.advanced_vertex], dim=0).cpu().numpy()
#         else:
#             all_vertices = mesh.vertices.cpu().numpy()
        
#         # 计算目标梯度限制
#         target_gradient = np.log(beta)
#         print(f"开始高阶点梯度控制优化，目标梯度: {target_gradient:.6f} (log({beta}))")
#         print(f"普通点数量: {n_base}, 高阶点数量: {n_advanced}")
#         print(f"初始平均尺寸 - 普通点: {np.mean(current_sizing[:n_base]):.6f}, 高阶点: {np.mean(current_sizing[n_base:]):.6f}")
        
#         for iteration in range(num_iterations):
#             new_sizing = current_sizing.copy()
            
#             # 只更新高阶点
#             for i in advanced_indices:
#                 if i not in adjacency_dict:
#                     continue
                    
#                 neighbors = adjacency_dict[i]
#                 if not neighbors:
#                     continue
                
#                 # 计算所有邻接点的约束尺寸
#                 constraint_sizes = []
                
#                 for j in neighbors:
#                     if j >= n_total:
#                         continue
                        
#                     # 计算两点间距离
#                     dist = np.linalg.norm(all_vertices[i] - all_vertices[j])
#                     if dist < 1e-10:
#                         constraint_sizes.append(current_sizing[j])
#                         continue
                    
#                     # 计算基于梯度约束的尺寸上限
#                     # s_j + l * gradient = s_j + dist * log(beta)
#                     constraint_size = current_sizing[j] + dist * target_gradient
#                     constraint_sizes.append(constraint_size)
                
#                 if constraint_sizes:
#                     # 取所有约束尺寸的最小值
#                     min_constraint = np.min(constraint_sizes)
                    
#                     # 更新高阶点尺寸：取当前尺寸和最小约束尺寸的最小值
#                     new_sizing[i] = min(current_sizing[i], min_constraint)
            
#             # 确保高阶点不会超过原始尺寸
#             new_sizing[advanced_indices] = np.minimum(
#                 new_sizing[advanced_indices], 
#                 original_sizing[advanced_indices]
#             )
            
#             # 应用最小尺寸限制
#             min_size = np.max(original_sizing) * 0.01
#             new_sizing = np.maximum(new_sizing, min_size)
            
#             # 保持普通特征点不变
#             if feature_mask is not None:
#                 feature_indices = torch.where(feature_mask)[0].cpu().numpy()
#                 for idx in feature_indices:
#                     if idx < n_base:  # 只处理普通特征点
#                         new_sizing[idx] = original_sizing[idx]
            
#             # 检查收敛
#             change_norm = np.linalg.norm(new_sizing - current_sizing)
#             current_sizing = new_sizing
            
#             # 计算当前梯度统计
#             avg_gradient = self._compute_average_gradient(
#                 current_sizing, all_vertices, adjacency_dict, advanced_indices
#             )
#             max_gradient = self._compute_max_gradient(
#                 current_sizing, all_vertices, adjacency_dict, advanced_indices
#             )
            
#             # if iteration % 10 == 0:
#             base_avg = np.mean(current_sizing[:n_base])
#             advanced_avg = np.mean(current_sizing[n_base:])
#             print(f"迭代 {iteration}: 普通点={base_avg:.6f}, 高阶点={advanced_avg:.6f}")
#             print(f"          平均梯度={avg_gradient:.6f}, 最大梯度={max_gradient:.6f}, 变化={change_norm:.6f}")
            
#             # if change_norm < self.tol:
#             #     print(f"收敛于迭代 {iteration}")
#             #     break
        
#         # 输出最终统计
#         final_stats = self._compute_final_statistics(
#             current_sizing, original_sizing, all_vertices, 
#             adjacency_dict, n_base, advanced_indices, target_gradient
#         )
        
#         # 拆分结果
#         base_sizing, advanced_sizing = self.split_unified_sizing_vector(
#             torch.tensor(current_sizing, dtype=torch.float32), mesh
#         )
        
#         return base_sizing, advanced_sizing, final_stats
    
#     def _compute_average_gradient(self, sizing, vertices, adjacency_dict, advanced_indices):
#         """计算高阶点的平均尺寸梯度"""
#         gradients = []
#         n_total = len(sizing)
        
#         for i in advanced_indices:
#             if i not in adjacency_dict:
#                 continue
                
#             neighbors = adjacency_dict[i]
#             for j in neighbors:
#                 if j >= n_total:
#                     continue
                    
#                 dist = np.linalg.norm(vertices[i] - vertices[j])
#                 if dist < 1e-10:
#                     continue
                
#                 if sizing[j] > 1e-10 and sizing[i] > 1e-10:
#                     size_ratio = sizing[i] / sizing[j]
#                     log_ratio = np.log(size_ratio)
#                     gradient = abs(log_ratio / dist)
#                     gradients.append(gradient)
        
#         return np.mean(gradients) if gradients else 0.0
    
#     def _compute_max_gradient(self, sizing, vertices, adjacency_dict, advanced_indices):
#         """计算高阶点的最大尺寸梯度"""
#         max_grad = 0.0
#         n_total = len(sizing)
        
#         for i in advanced_indices:
#             if i not in adjacency_dict:
#                 continue
                
#             neighbors = adjacency_dict[i]
#             for j in neighbors:
#                 if j >= n_total:
#                     continue
                    
#                 dist = np.linalg.norm(vertices[i] - vertices[j])
#                 if dist < 1e-10:
#                     continue
                
#                 if sizing[j] > 1e-10 and sizing[i] > 1e-10:
#                     # size_ratio = sizing[i] / sizing[j]
#                     # log_ratio = np.log(size_ratio)
#                     # gradient = abs(log_ratio / dist)
#                     gradient = np.log(abs(sizing[i] - sizing[j]) / dist + 1e-6)
#                     max_grad = max(max_grad, gradient)
        
#         return max_grad
    
#     def _compute_final_statistics(self, final_sizing, original_sizing, vertices, 
#                                 adjacency_dict, n_base, advanced_indices, target_gradient):
#         """计算最终统计信息"""
#         final_base_avg = np.mean(final_sizing[:n_base])
#         final_advanced_avg = np.mean(final_sizing[n_base:])
#         original_base_avg = np.mean(original_sizing[:n_base])
#         original_advanced_avg = np.mean(original_sizing[n_base:])
        
#         base_reduction = (original_base_avg - final_base_avg) / original_base_avg * 100
#         advanced_reduction = (original_advanced_avg - final_advanced_avg) / original_advanced_avg * 100
        
#         avg_gradient = self._compute_average_gradient(
#             final_sizing, vertices, adjacency_dict, advanced_indices
#         )
#         max_gradient = self._compute_max_gradient(
#             final_sizing, vertices, adjacency_dict, advanced_indices
#         )
        
#         stats = {
#             'original_base_avg': original_base_avg,
#             'final_base_avg': final_base_avg,
#             'base_reduction_pct': base_reduction,
#             'original_advanced_avg': original_advanced_avg,
#             'final_advanced_avg': final_advanced_avg,
#             'advanced_reduction_pct': advanced_reduction,
#             'final_avg_gradient': avg_gradient,
#             'final_max_gradient': max_gradient,
#             'target_gradient': target_gradient,
#             'gradient_achieved': max_gradient <= target_gradient
#         }
        
#         print("优化完成统计:")
#         print(f"  普通点: {original_base_avg:.6f} -> {final_base_avg:.6f} (变化 {base_reduction:+.2f}%)")
#         print(f"  高阶点: {original_advanced_avg:.6f} -> {final_advanced_avg:.6f} (缩小 {advanced_reduction:.2f}%)")
#         print(f"  最终平均梯度: {avg_gradient:.6f}")
#         print(f"  最终最大梯度: {max_gradient:.6f}")
#         print(f"  目标梯度: {target_gradient:.6f}")
#         print(f"  梯度目标{'已达成' if stats['gradient_achieved'] else '未达成'}")
        
#         return stats

#     # 保留原有的邻接关系构建方法
#     def build_unified_adjacency(self, mesh):
#         """
#         构建统一邻接关系：普通点 + 高阶点
#         """
#         n_base = len(mesh.vertices)
#         n_advanced = len(mesh.advanced_vertex) if mesh.advanced_vertex is not None else 0
#         n_total = n_base + n_advanced
        
#         adjacency_dict = defaultdict(list)
        
#         # 步骤1: 构建普通点之间的基础邻接
#         base_adjacency = self._build_base_adjacency(mesh)
#         for i, neighbors in base_adjacency.items():
#             adjacency_dict[i].extend(neighbors)
        
#         if n_advanced > 0:
#             # 步骤2: 普通点与邻接边上的高阶点连接
#             self._connect_base_to_advanced(mesh, adjacency_dict, n_base)
            
#             # 步骤3: 高阶点与所在边的顶点连接  
#             self._connect_advanced_to_base(mesh, adjacency_dict, n_base)
            
#             # 步骤4: 高阶点与邻接面上的所有点连接
#             self._connect_advanced_to_advanced(mesh, adjacency_dict, n_base)
        
#         return dict(adjacency_dict), n_total

#     def create_unified_sizing_vector(self, mesh):
#         """创建统一的尺寸向量"""
#         n_base = len(mesh.vertices)
#         n_advanced = len(mesh.advanced_vertex) if mesh.advanced_vertex is not None else 0
#         n_total = n_base + n_advanced
        
#         unified_sizing = np.zeros(n_total)
        
#         # 填充普通点尺寸
#         if mesh.sizing_values is not None:
#             base_sizing = mesh.sizing_values.cpu().numpy()
#             unified_sizing[:n_base] = base_sizing[:, 0]
#         else:
#             unified_sizing[:n_base] = 1.0
        
#         # 填充高阶点尺寸
#         if n_advanced > 0 and mesh.advanced_size is not None:
#             advanced_sizing = mesh.advanced_size.cpu().numpy()
#             unified_sizing[n_base:n_total] = advanced_sizing[:, 0]
#         elif n_advanced > 0:
#             # 如果没有高阶尺寸，从相邻普通点插值
#             for adv_idx in range(n_advanced):
#                 global_idx = n_base + adv_idx
#                 edge_idx = mesh.advanced_vertex_edge_map[adv_idx].item()
#                 if edge_idx >= 0:
#                     v1, v2 = mesh.edges[edge_idx].cpu().numpy()
#                     unified_sizing[global_idx] = 0.5 * (unified_sizing[v1] + unified_sizing[v2])
        
#         return torch.tensor(unified_sizing, dtype=torch.float32)


class ShrinkingLaplaceOptimizer(UnifiedLaplaceOptimizer):
    """基于边梯度控制的高阶点尺寸平滑优化器"""
    
    def optimize_advanced_gradient_control(self, mesh, beta=1.5, feature_mask=None, max_iterations=50):
        """
        基于边梯度控制的高阶点尺寸平滑优化
        
        参数:
        mesh: 网格对象
        beta: 梯度控制参数，梯度控制在 log(beta) 范围内
        feature_mask: 特征点掩码
        max_iterations: 最大迭代次数
        """
        # 获取网格基本信息
        n_base = len(mesh.vertices)
        n_advanced = len(mesh.advanced_vertex) if mesh.advanced_vertex is not None else 0
        n_total = n_base + n_advanced
        
        # 创建统一的尺寸向量
        unified_sizing = self.create_unified_sizing_vector(mesh)
        current_sizing = unified_sizing.cpu().numpy()
        original_sizing = current_sizing.copy()
        
        # 构建边列表和边长度
        edges, edge_lengths = self._build_advanced_edges(mesh)
        num_edges = len(edges)
        
        # 计算目标梯度限制
        ln_beta = np.log(beta)
        
        # 收敛参数
        resolution = max(1.1 / num_edges, 1e-5)
        tolerant_max_times = max(5, int(0.05 * max_iterations))
        tolerant_count = 0
        
        print(f"开始高阶点梯度控制优化，目标梯度: {ln_beta:.6f} (log({beta}))")
        print(f"普通点数量: {n_base}, 高阶点数量: {n_advanced}, 边数量: {num_edges}")
        
        for iteration in range(max_iterations):
            changed = 0
            
            # 遍历所有边
            for j in range(num_edges):
                n0, n1 = edges[j]
                length = edge_lengths[j]
                
                # 计算尺寸差
                delta_s = abs(current_sizing[n0] - current_sizing[n1])
                
                # 检查梯度是否超过限制
                if (delta_s / length) > ln_beta:
                    if current_sizing[n0] > current_sizing[n1]:
                        # 调整n0的尺寸
                        new_size = current_sizing[n1] + ln_beta * length
                        # 确保不会超过原始尺寸（只缩小）
                        new_size = min(new_size, current_sizing[n0], original_sizing[n0])
                        # 只更新高阶点
                        if n0 >= n_base:  # n0是高阶点
                            current_sizing[n0] = new_size
                            changed += 1
                    else:
                        # 调整n1的尺寸
                        new_size = current_sizing[n0] + ln_beta * length
                        # 确保不会超过原始尺寸（只缩小）
                        new_size = min(new_size, current_sizing[n1], original_sizing[n1])
                        # 只更新高阶点
                        if n1 >= n_base:  # n1是高阶点
                            current_sizing[n1] = new_size
                            changed += 1
            
            # 保持普通特征点不变
            # if feature_mask is not None:
            #     feature_indices = torch.where(feature_mask)[0].cpu().numpy()
            #     for idx in feature_indices:
            #         if idx < n_base:  # 只处理普通特征点
            #             current_sizing[idx] = original_sizing[idx]
            
            # 收敛检查
            change_ratio = changed / num_edges
            if change_ratio <= resolution:
                tolerant_count += 1
                print(f"迭代 {iteration}: 满足收敛条件 {tolerant_count}/{tolerant_max_times}")
            else:
                tolerant_count = 0
            
            # 计算当前梯度统计
            if iteration % 10 == 0 or tolerant_count > 0:
                avg_gradient, max_gradient = self._compute_gradient_statistics(
                    current_sizing, edges, edge_lengths, n_base
                )
                base_avg = np.mean(current_sizing[:n_base])
                advanced_avg = np.mean(current_sizing[n_base:])
                print(f"迭代 {iteration}: 普通点={base_avg:.6f}, 高阶点={advanced_avg:.6f}")
                print(f"          平均梯度={avg_gradient:.6f}, 最大梯度={max_gradient:.6f}, 变化边数={changed}/{num_edges}")
            
            if tolerant_count >= tolerant_max_times:
                print(f"收敛于迭代 {iteration}")
                break
        
        # 输出最终统计
        final_stats = self._compute_final_statistics(
            current_sizing, original_sizing, edges, edge_lengths, n_base, ln_beta
        )
        
        # 拆分结果
        base_sizing, advanced_sizing = self.split_unified_sizing_vector(
            torch.tensor(current_sizing, dtype=torch.float32), mesh
        )
        
        return base_sizing, advanced_sizing, final_stats
    
    def _build_advanced_edges(self, mesh):
        """
        构建包含高阶点的边列表和边长度
        
        返回:
        edges: 边列表，每个元素是 (顶点索引1, 顶点索引2)
        edge_lengths: 对应的边长度列表
        """
        n_base = len(mesh.vertices)
        n_advanced = len(mesh.advanced_vertex) if mesh.advanced_vertex is not None else 0
        n_total = n_base + n_advanced
        
        # 获取所有顶点坐标
        if mesh.advanced_vertex is not None:
            all_vertices = torch.cat([mesh.vertices, mesh.advanced_vertex], dim=0).cpu().numpy()
        else:
            all_vertices = mesh.vertices.cpu().numpy()
        
        # 构建邻接关系
        adjacency_dict, _ = self.build_unified_adjacency(mesh)
        
        # 收集所有包含高阶点的边（避免重复）
        edge_set = set()
        edges = []
        
        # 遍历所有点，只考虑与高阶点相关的边
        for i in range(n_total):
            # 如果当前点是高阶点，或者邻接点中有高阶点
            if i >= n_base or any(neighbor >= n_base for neighbor in adjacency_dict.get(i, [])):
                for neighbor in adjacency_dict.get(i, []):
                    if neighbor < n_total:  # 确保邻接点在有效范围内
                        # 创建有序边对，避免重复
                        edge = (min(i, neighbor), max(i, neighbor))
                        if edge not in edge_set:
                            edge_set.add(edge)
                            edges.append(edge)
        
        # 计算边长度
        edge_lengths = []
        for edge in edges:
            v0, v1 = edge
            length = np.linalg.norm(all_vertices[v0] - all_vertices[v1])
            edge_lengths.append(length)
        
        return edges, edge_lengths
    
    def _compute_gradient_statistics(self, sizing, edges, edge_lengths, n_base):
        """计算梯度统计信息"""
        gradients = []
        
        for j, (n0, n1) in enumerate(edges):
            length = edge_lengths[j]
            if length < 1e-10:
                continue
                
            delta_s = abs(sizing[n0] - sizing[n1])
            gradient = delta_s / length
            gradients.append(gradient)
        
        if gradients:
            avg_gradient = np.mean(gradients)
            max_gradient = np.max(gradients)
        else:
            avg_gradient = 0.0
            max_gradient = 0.0
            
        return avg_gradient, max_gradient
    
    def _compute_final_statistics(self, final_sizing, original_sizing, edges, edge_lengths, n_base, target_gradient):
        """计算最终统计信息"""
        final_base_avg = np.mean(final_sizing[:n_base])
        final_advanced_avg = np.mean(final_sizing[n_base:])
        original_base_avg = np.mean(original_sizing[:n_base])
        original_advanced_avg = np.mean(original_sizing[n_base:])
        
        base_reduction = (original_base_avg - final_base_avg) / original_base_avg * 100
        advanced_reduction = (original_advanced_avg - final_advanced_avg) / original_advanced_avg * 100
        
        avg_gradient, max_gradient = self._compute_gradient_statistics(
            final_sizing, edges, edge_lengths, n_base
        )
        
        # 检查有多少边满足梯度约束
        satisfied_edges = 0
        total_edges = len(edges)
        
        for j, (n0, n1) in enumerate(edges):
            length = edge_lengths[j]
            if length < 1e-10:
                satisfied_edges += 1
                continue
                
            delta_s = abs(final_sizing[n0] - final_sizing[n1])
            gradient = delta_s / length
            if gradient <= target_gradient:
                satisfied_edges += 1
        
        satisfaction_ratio = satisfied_edges / total_edges if total_edges > 0 else 1.0
        
        stats = {
            'original_base_avg': original_base_avg,
            'final_base_avg': final_base_avg,
            'base_reduction_pct': base_reduction,
            'original_advanced_avg': original_advanced_avg,
            'final_advanced_avg': final_advanced_avg,
            'advanced_reduction_pct': advanced_reduction,
            'final_avg_gradient': avg_gradient,
            'final_max_gradient': max_gradient,
            'target_gradient': target_gradient,
            'satisfaction_ratio': satisfaction_ratio,
            'gradient_achieved': satisfaction_ratio >= 0.99  # 99%的边满足条件
        }
        
        print("优化完成统计:")
        print(f"  普通点: {original_base_avg:.6f} -> {final_base_avg:.6f} (变化 {base_reduction:+.2f}%)")
        print(f"  高阶点: {original_advanced_avg:.6f} -> {final_advanced_avg:.6f} (缩小 {advanced_reduction:.2f}%)")
        print(f"  最终平均梯度: {avg_gradient:.6f}")
        print(f"  最终最大梯度: {max_gradient:.6f}")
        print(f"  目标梯度: {target_gradient:.6f}")
        print(f"  梯度约束满足率: {satisfaction_ratio:.2%}")
        print(f"  梯度目标{'已达成' if stats['gradient_achieved'] else '未达成'}")
        
        return stats

    # 保留原有的邻接关系构建方法
    def build_unified_adjacency(self, mesh):
        """
        构建统一邻接关系：普通点 + 高阶点
        """
        n_base = len(mesh.vertices)
        n_advanced = len(mesh.advanced_vertex) if mesh.advanced_vertex is not None else 0
        n_total = n_base + n_advanced
        
        adjacency_dict = defaultdict(list)
        
        # 步骤1: 构建普通点之间的基础邻接
        base_adjacency = self._build_base_adjacency(mesh)
        for i, neighbors in base_adjacency.items():
            adjacency_dict[i].extend(neighbors)
        
        if n_advanced > 0:
            # 步骤2: 普通点与邻接边上的高阶点连接
            self._connect_base_to_advanced(mesh, adjacency_dict, n_base)
            
            # 步骤3: 高阶点与所在边的顶点连接  
            self._connect_advanced_to_base(mesh, adjacency_dict, n_base)
            
            # 步骤4: 高阶点与邻接面上的所有点连接
            self._connect_advanced_to_advanced(mesh, adjacency_dict, n_base)
        
        return dict(adjacency_dict), n_total

    def create_unified_sizing_vector(self, mesh):
        """创建统一的尺寸向量"""
        n_base = len(mesh.vertices)
        n_advanced = len(mesh.advanced_vertex) if mesh.advanced_vertex is not None else 0
        n_total = n_base + n_advanced
        
        unified_sizing = np.zeros(n_total)
        
        # 填充普通点尺寸
        if mesh.sizing_values is not None:
            base_sizing = mesh.sizing_values.cpu().numpy()
            unified_sizing[:n_base] = base_sizing[:, 0]
        else:
            unified_sizing[:n_base] = 1.0
        
        # 填充高阶点尺寸
        if n_advanced > 0 and mesh.advanced_size is not None:
            advanced_sizing = mesh.advanced_size.cpu().numpy()
            unified_sizing[n_base:n_total] = advanced_sizing[:, 0]
        elif n_advanced > 0:
            # 如果没有高阶尺寸，从相邻普通点插值
            for adv_idx in range(n_advanced):
                global_idx = n_base + adv_idx
                edge_idx = mesh.advanced_vertex_edge_map[adv_idx].item()
                if edge_idx >= 0:
                    v1, v2 = mesh.edges[edge_idx].cpu().numpy()
                    unified_sizing[global_idx] = 0.5 * (unified_sizing[v1] + unified_sizing[v2])
        
        return torch.tensor(unified_sizing, dtype=torch.float32)