import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class QuadraticFunctionSolver:
    def __init__(self, start_point, end_point, max_gradient, max_value, intermediate_points=None):
        """
        初始化二次函数求解器
        
        参数:
        start_point: (x, y) 起点坐标
        end_point: (x, y) 终点坐标
        max_gradient: 最大梯度（绝对值）
        max_value: 函数最大值
        intermediate_points: 中间点列表 [(x1, y1), (x2, y2), ...]
        """
        self.start_point = start_point
        self.end_point = end_point
        self.max_gradient = max_gradient
        self.max_value = max_value
        self.intermediate_points = intermediate_points if intermediate_points else []
        
    def quadratic_function(self, x, coeffs):
        """二次函数: y = ax² + bx + c"""
        a, b, c = coeffs
        return a * x**2 + b * x + c
    
    def gradient_function(self, x, coeffs):
        """梯度函数: dy/dx = 2ax + b"""
        a, b, _ = coeffs
        return 2 * a * x + b
    
    def objective_function(self, coeffs):
        """目标函数：最小化函数值与约束点的误差平方和"""
        error = 0.0
        
        # 起点约束
        x_start, y_start = self.start_point
        y_pred = self.quadratic_function(x_start, coeffs)
        error += (y_pred - y_start)**2
        
        # 终点约束
        x_end, y_end = self.end_point
        y_pred = self.quadratic_function(x_end, coeffs)
        error += (y_pred - y_end)**2
        
        # 中间点约束
        for point in self.intermediate_points:
            x, y = point
            y_pred = self.quadratic_function(x, coeffs)
            error += (y_pred - y)**2
            
        return error
    
    def constraints(self):
        """定义约束条件"""
        constraints_list = []
        
        # 梯度约束
        # def gradient_constraint(coeffs):
        #     a, b, _ = coeffs
            
        #     # 检查起点和终点的梯度
        #     grad_start = self.gradient_function(self.start_point[0], coeffs)
        #     grad_end = self.gradient_function(self.end_point[0], coeffs)
            
        #     # 梯度绝对值不能超过最大值
        #     return self.max_gradient - max(abs(grad_start), abs(grad_end))
        def gradient_constraint(coeffs):
            a, b, _ = coeffs
            x_start, x_end = self.start_point[0], self.end_point[0]
            
            # 二次函数的梯度是线性的，最大值出现在端点
            grad_start = self.gradient_function(x_start, coeffs)
            grad_end = self.gradient_function(x_end, coeffs)
            
            # 检查所有端点的梯度
            max_grad = max(abs(grad_start), abs(grad_end))
            
            # 如果有中间点，也检查中间点的梯度
            for point in self.intermediate_points:
                x, _ = point
                grad = self.gradient_function(x, coeffs)
                max_grad = max(max_grad, abs(grad))
            
            return self.max_gradient - max_grad
        
        # 最大值约束
        # def max_value_constraint(coeffs):
        #     a, b, c = coeffs
            
        #     # 对于二次函数，最大值出现在顶点或端点
        #     x_start, y_start = self.start_point
        #     x_end, y_end = self.end_point
            
        #     # 计算顶点位置（如果存在）
        #     if a != 0:
        #         vertex_x = -b / (2 * a)
        #         # 如果顶点在区间内，检查顶点值
        #         if min(x_start, x_end) <= vertex_x <= max(x_start, x_end):
        #             vertex_y = self.quadratic_function(vertex_x, coeffs)
        #             return self.max_value - max(y_start, y_end, vertex_y)
            
        #     return self.max_value - max(y_start, y_end)
        def max_value_constraint(coeffs):
            a, b, c = coeffs
            x_start, x_end = self.start_point[0], self.end_point[0]
            
            # 检查所有已知点的函数值
            points_to_check = [self.start_point, self.end_point] + self.intermediate_points
            max_y = max(self.quadratic_function(x, coeffs) for x, y in points_to_check)
            
            # 检查顶点（如果存在且在区间内）
            if a != 0:
                vertex_x = -b / (2 * a)
                if min(x_start, x_end) <= vertex_x <= max(x_start, x_end):
                    vertex_y = self.quadratic_function(vertex_x, coeffs)
                    max_y = max(max_y, vertex_y)
            
            return self.max_value - max_y
        
        constraints_list.append({'type': 'ineq', 'fun': gradient_constraint})
        constraints_list.append({'type': 'ineq', 'fun': max_value_constraint})
        
        return constraints_list
    
    def solve(self):
        """求解二次函数系数"""
        # 初始猜测：线性函数
        x_start, y_start = self.start_point
        x_end, y_end = self.end_point
        
        # 初始线性近似
        if x_end != x_start:
            slope = (y_end - y_start) / (x_end - x_start)
        else:
            slope = 0
            
        # 初始系数：接近线性函数
        initial_guess = [0.001, slope, y_start - slope * x_start]
        
        # 设置边界条件
        bounds = [(-10, 10), (-10, 10), (-100, 100)]
        
        # 优化求解
        result = minimize(
            self.objective_function,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=self.constraints(),
            options={'ftol': 1e-4, 'disp': False, 'maxiter': 10000}
        )
        
        if result.success:
            return result.x
        else:
            raise ValueError("无法找到满足条件的二次函数")
    
    def evaluate(self, coeffs, x):
        """在给定点x处评估函数值"""
        return self.quadratic_function(x, coeffs)
    
    def plot_function(self, coeffs, num_points=100):
        """绘制函数图像"""
        x_start, y_start = self.start_point
        x_end, y_end = self.end_point
        
        x_values = np.linspace(min(x_start, x_end) - 1, max(x_start, x_end) + 1, num_points)
        y_values = self.quadratic_function(x_values, coeffs)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, 'b-', label='二次函数')
        plt.plot(x_start, y_start, 'ro', markersize=8, label='起点')
        plt.plot(x_end, y_end, 'go', markersize=8, label='终点')
        
        for i, point in enumerate(self.intermediate_points):
            x, y = point
            plt.plot(x, y, 'mo', markersize=6, label=f'中间点{i+1}')
        
        # 添加梯度信息
        grad_start = self.gradient_function(x_start, coeffs)
        grad_end = self.gradient_function(x_end, coeffs)
        
        plt.axhline(y=self.max_value, color='r', linestyle='--', alpha=0.7, label=f'最大值约束: {self.max_value}')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'二次函数: y = {coeffs[0]:.3f}x² + {coeffs[1]:.3f}x + {coeffs[2]:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("result.png")
        plt.show()
        
        print(f"起点梯度: {grad_start:.3f}")
        print(f"终点梯度: {grad_end:.3f}")
        print(f"最大梯度限制: ±{self.max_gradient}")

    def verify_constraints(self, coeffs):
        """验证所有约束条件是否满足"""
        a, b, c = coeffs
        violations = []
        
        # 1. 验证起点和终点
        x_start, y_start = self.start_point
        y_pred_start = self.quadratic_function(x_start, coeffs)
        if abs(y_pred_start - y_start) > 1e-3:
            violations.append(f"起点约束不满足: 期望 {y_start}, 得到 {y_pred_start}")
        
        x_end, y_end = self.end_point
        y_pred_end = self.quadratic_function(x_end, coeffs)
        if abs(y_pred_end - y_end) > 1e-3:
            violations.append(f"终点约束不满足: 期望 {y_end}, 得到 {y_pred_end}")
        
        # 2. 验证中间点
        for i, point in enumerate(self.intermediate_points):
            x, y = point
            y_pred = self.quadratic_function(x, coeffs)
            if abs(y_pred - y) > 1e-3:
                violations.append(f"中间点 {i+1} 约束不满足: 期望 {y}, 得到 {y_pred}")
        
        # 3. 验证梯度约束
        points_to_check = [self.start_point, self.end_point] + self.intermediate_points
        max_gradient_found = 0
        for point in points_to_check:
            x, _ = point
            grad = abs(self.gradient_function(x, coeffs))
            max_gradient_found = max(max_gradient_found, grad)
        
        if max_gradient_found > self.max_gradient + 1e-3:
            violations.append(f"梯度约束不满足: 最大梯度 {max_gradient_found:.4f} > {self.max_gradient}")
        
        # 4. 验证最大值约束
        x_start, x_end = self.start_point[0], self.end_point[0]
        x_min, x_max = min(x_start, x_end), max(x_start, x_end)
        
        # 检查已知点和顶点
        points_to_check = [(x, self.quadratic_function(x, coeffs)) for x, _ in [self.start_point, self.end_point] + self.intermediate_points]
        
        if a != 0:
            vertex_x = -b / (2 * a)
            if x_min <= vertex_x <= x_max:
                vertex_y = self.quadratic_function(vertex_x, coeffs)
                points_to_check.append((vertex_x, vertex_y))
        
        max_value_found = max(y for x, y in points_to_check)
        if max_value_found > self.max_value + 1e-3:
            violations.append(f"最大值约束不满足: 最大值 {max_value_found:.4f} > {self.max_value}")
        
        return violations

def main():
    """示例使用"""
    # 示例1：基本用法（只有起点和终点）
    print("示例1：基本用法")
    solver1 = QuadraticFunctionSolver(
        start_point=(0, 0),
        end_point=(10, 5),
        max_gradient=1.0,
        max_value=10
    )
    
    try:
        coeffs1 = solver1.solve()
        print(f"二次函数系数: a={coeffs1[0]:.4f}, b={coeffs1[1]:.4f}, c={coeffs1[2]:.4f}")
        solver1.plot_function(coeffs1)
    except ValueError as e:
        print(f"错误: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # 示例2：带中间点
    print("示例2：带中间点")
    solver2 = QuadraticFunctionSolver(
        start_point=(0, 0),
        end_point=(10, 8),
        max_gradient=1.5,
        max_value=12,
        intermediate_points=[(4, 3), (7, 6)]
    )
    
    try:
        coeffs2 = solver2.solve()
        print(f"二次函数系数: a={coeffs2[0]:.4f}, b={coeffs2[1]:.4f}, c={coeffs2[2]:.4f}")
        solver2.plot_function(coeffs2)
    except ValueError as e:
        print(f"错误: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # 示例3：严格约束
    print("示例3：严格约束")
    solver3 = QuadraticFunctionSolver(
        start_point=(0, 0),
        end_point=(5, 2),
        max_gradient=0.5,
        max_value=3,
        intermediate_points=[(2, 1)]
    )
    
    try:
        coeffs3 = solver3.solve()
        print(f"二次函数系数: a={coeffs3[0]:.4f}, b={coeffs3[1]:.4f}, c={coeffs3[2]:.4f}")
        solver3.plot_function(coeffs3)
    except ValueError as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()