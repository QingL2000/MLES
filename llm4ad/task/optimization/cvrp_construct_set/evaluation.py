

from __future__ import annotations
import sys

sys.path.append('../../')
import copy
from typing import Any
import traceback
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy

from llm4ad.base import Evaluation
from llm4ad.task.optimization.cvrp_construct_set.get_instance import GetData
from llm4ad.task.optimization.cvrp_construct_set.template import template_program, task_description
import pickle
from typing import Optional, Tuple, List, Any, Set
from io import BytesIO
import base64
import json


class CVRPSEvaluation(Evaluation):
    def __init__(self,
                 timeout_seconds=60,
                 training_datasets=None,
                 testing_datasets=None,
                 whocall='mmeoh',
                 objective_value=0,
                 multimodal_Flag=True,
                 **kwargs):

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self.whocall = whocall
        self._mode = kwargs.get('run_mode', 'Training')
        self.final_objective_score = objective_value

        self._training_instances_number = 0
        self.instance_set = {}

        self._ins_cluster = {}  # Dict[int, Set[Any]]
        self._cluster_filename_map = {}
        cluster_id = 0

        for training_dataset in training_datasets:
            self._ins_cluster[cluster_id] = set()
            self._cluster_filename_map[cluster_id] = training_dataset

            loaded_instances = pickle.load(open(training_dataset, 'rb'))
            for k, v in loaded_instances.items():
                self.instance_set[self._training_instances_number] = v
                self._ins_cluster[cluster_id].add(self._training_instances_number)
                self._training_instances_number += 1
            cluster_id += 1

        self.instance_id_set = tuple(self.instance_set.keys())

        self._testing_instances_number = 0
        self.ins_to_be_solve_set = {}

        self._test_ins_cluster = {}  # Dict[int, Set[Any]]
        self._test_cluster_filename_map = {}
        test_cluster_id = 0

        for testing_dataset in testing_datasets:
            self._test_ins_cluster[test_cluster_id] = set()
            self._test_cluster_filename_map[test_cluster_id] = testing_dataset
            loaded_instances = pickle.load(open(testing_dataset, 'rb'))
            for k, v in loaded_instances.items():
                self.ins_to_be_solve_set[self._testing_instances_number] = v
                self._test_ins_cluster[test_cluster_id].add(self._testing_instances_number)
                self._testing_instances_number += 1
            test_cluster_id += 1

        self.to_be_solve_instance_id_set = tuple(self.ins_to_be_solve_set.keys())

        self.n_instance = 100000
        self.problem_size = 0
        self.multimodal_Flag = multimodal_Flag

        if self._mode == 'Training' and not self.instance_set:
            # 使用更标准的 Python 异常处理
            raise ValueError("没有提供Training实例集 (instance_set)，无法进行评估。")

        if self._mode == 'Using' and not self.ins_to_be_solve_set:
            # 使用更标准的 Python 异常处理
            raise ValueError("没有提供Testing实例集 (ins_to_be_solve_set)，无法进行评估。")

        if self._mode == 'Combined' and (not self.instance_set or not self.ins_to_be_solve_set):
            # 使用更标准的 Python 异常处理
            raise ValueError("缺少Training或Testing实例集 ，无法进行评估。")

    def plot_clean_solution(self, instance: np.ndarray, route: list, demands: list, vehicle_capacity: int):
        """
        A cleaner visualization for CVRP diagnosis.
        Focuses on geometry, clustering, and node importance (demand).
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Extract coordinates
        x = instance[:, 0]
        y = instance[:, 1]

        # Calculate metrics for title
        total_distance = 0

        # Split routes
        routes = []
        current_route = []
        route_loads = []
        current_load = 0

        # Reconstruct routes and calculate loads
        # Note: route list usually looks like [0, 5, 2, 0, 3, 4, 0] or similar
        # We need to handle the specific format from your heuristic

        # Standardize route format: ensure it starts/ends with 0 if not implicit
        # Your heuristic returns a single list. Let's parse it carefully.

        temp_route = [0]
        for node in route:
            if node == 0:
                if len(temp_route) > 1:  # Finish current route
                    temp_route.append(0)
                    routes.append(temp_route)
                    route_loads.append(current_load)
                    temp_route = [0]
                    current_load = 0
            else:
                temp_route.append(node)
                current_load += demands[node]
        if len(temp_route) > 1:  # Catch the last route if it doesn't end with 0
            temp_route.append(0)
            routes.append(temp_route)
            route_loads.append(current_load)

        # Calculate Distance
        for r in routes:
            for i in range(len(r) - 1):
                dist = np.linalg.norm(instance[r[i]] - instance[r[i + 1]])
                total_distance += dist

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # 1. Plot Customer Nodes (Size proportional to demand)
        # Scale size: Min size 20, Max size 200 based on demand ratio
        demand_sizes = [20 + (d / vehicle_capacity) * 300 for d in demands]

        # Plot Depot (Red Square)
        ax.scatter(x[0], y[0], c='red', s=150, marker='s', zorder=10, label='Depot')

        # Plot Customers (Blue Circles, sized by demand)
        # We slice [1:] to skip depot in demand/coordinate lists for scatter
        ax.scatter(x[1:], y[1:], c='steelblue', s=demand_sizes[1:], alpha=0.6, edgecolors='k', zorder=5,
                   label='Customers (Size ~ Demand)')

        # 2. Plot Routes
        colors = plt.cm.get_cmap('tab10', len(routes))  # Use tab20 for more variety if needed

        for idx, r in enumerate(routes):
            r_indices = r
            r_x = x[r_indices]
            r_y = y[r_indices]

            # Plot the path
            ax.plot(r_x, r_y, color=colors(idx), linewidth=1.5, alpha=0.8, zorder=1)

            # Optional: Mark the "farthest" point or direction? No, keep it clean.

        # 3. Informative Title (The Dashboard)
        avg_fill_rate = np.mean([l / vehicle_capacity for l in route_loads]) * 100
        title_text = (
            f"CVRP Diagnosis View\n"
            f"Vehicles: {len(routes)} | Avg Utilization: {avg_fill_rate:.1f}%"
        )

        ax.set_title(title_text, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_solution(self, instance: np.ndarray, route: list, demands: list, vehicle_capacity: int):
        """
        Plot the solution of the Capacitated Vehicle Routing Problem (CVRP).

        Args:
            instance: A 2D array of node coordinates (including the depot).
            route: A list representing the sequence of nodes visited in the route.
            demands: A list of demands for each node.
            vehicle_capacity: The capacity of the vehicle.
        """
        # Extract coordinates
        x = instance[:, 0]
        y = instance[:, 1]

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot depot (node 0)
        ax.plot(x[0], y[0], 'ro', markersize=10, label='Depot')
        ax.text(x[0], y[0], 'Depot', ha='center', va='bottom', fontsize=12)

        # Plot customer nodes
        for i in range(1, len(x)):
            ax.plot(x[i], y[i], 'bo', markersize=8)
            ax.text(x[i], y[i], f'C{i}\nDem: {demands[i]}', ha='center', va='bottom', fontsize=8)

        # Split the route into individual vehicle routes based on depot visits
        routes = []
        current_route = []
        for node in route:
            current_route.append(node)
            if node == 0 and len(current_route) > 1:  # End of a route (return to depot)
                routes.append(current_route)
                current_route = [0]  # Start a new route from the depot
        if current_route:  # Add the last route if it exists
            routes.append(current_route)

        # Plot each route in a different color
        colors = plt.cm.tab10.colors  # Use a colormap for distinct colors
        for i, r in enumerate(routes):
            color = colors[i % len(colors)]  # Cycle through colors
            for j in range(len(r) - 1):
                start_node = r[j]
                end_node = r[j + 1]
                ax.plot([x[start_node], x[end_node]], [y[start_node], y[end_node]], color=color, linestyle='-',
                        linewidth=1, label=f'Route {i + 1}' if j == 0 else None)

                # Add load information
                if end_node != 0:  # If not returning to the depot
                    ax.text((x[start_node] + x[end_node]) / 2, (y[start_node] + y[end_node]) / 2,
                            f'Load: {sum(demands[r[:j + 1]])}', ha='center', va='center', fontsize=8, rotation=45)

            # Mark start and end nodes of the route with triangles (excluding depot)
            if len(r) > 1:
                ax.plot(x[r[1]], y[r[1]], '^', color=color, markersize=10,
                        label='Start' if i == 0 else None)  # Start node
                ax.plot(x[r[-2]], y[r[-2]], 'v', color=color, markersize=10,
                        label='End' if i == 0 else None)  # End node

        # Set axis labels and title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Capacitated Vehicle Routing Problem (CVRP) Solution')
        ax.legend(loc='upper right')

        # Show the plot
        plt.tight_layout()
        plt.show()

    def tour_cost(self, instance, solution):
        cost = 0
        for j in range(len(solution) - 1):
            cost += np.linalg.norm(instance[int(solution[j])] - instance[int(solution[j + 1])])
        cost += np.linalg.norm(instance[int(solution[-1])] - instance[int(solution[0])])
        return cost

    def route_construct(self, distance_matrix, demands, vehicle_capacity, heuristic):
        route = []
        current_load = 0
        current_node = 0
        route.append(current_node)

        unvisited_nodes = set(range(1, self.problem_size))  # Assuming node 0 is the depot
        all_nodes = np.array(list(unvisited_nodes))
        feasible_unvisited_nodes = all_nodes

        while unvisited_nodes:
            next_node = heuristic(current_node,
                                  0,
                                  feasible_unvisited_nodes,  # copy
                                  vehicle_capacity - current_load,
                                  copy.deepcopy(demands),  # copy
                                  copy.deepcopy(distance_matrix))  # copy
            if next_node == 0:
                # Update route and load
                route.append(next_node)
                current_load = 0
                current_node = 0
            else:
                # Update route and load
                route.append(next_node)
                current_load += demands[next_node]
                unvisited_nodes.remove(next_node)
                current_node = next_node

            feasible_nodes_capacity = np.array(
                [node for node in all_nodes if current_load + demands[node] <= vehicle_capacity])
            # Determine feasible and unvisited nodes
            feasible_unvisited_nodes = np.intersect1d(feasible_nodes_capacity, list(unvisited_nodes))

            if len(unvisited_nodes) > 0 and len(feasible_unvisited_nodes) < 1:
                route.append(0)
                current_load = 0
                current_node = 0
                feasible_unvisited_nodes = np.array(list(unvisited_nodes))

        # check if not all nodes have been visited 
        independent_values = set(route)
        if len(independent_values) != self.problem_size:
            return None
        return route

    def evaluate_single(self, heuristic: callable, instance_information=None):
        # instance, distance_matrix, demands, vehicle_capacity, baseline = instance_information
        instance = instance_information['coordinates']
        distance_matrix = instance_information['distances']
        demands = instance_information['demands']
        vehicle_capacity = instance_information['capacity']

        baseline = instance_information['label']
        self.problem_size = len(demands)
        route = self.route_construct(distance_matrix, demands, vehicle_capacity, heuristic)
        instance_info = {'coordinates': instance,
                         'distances': distance_matrix,
                         'demands': demands,
                         'vehicle_capacity': vehicle_capacity,
                         'route': route,}
        LLM_dis = self.tour_cost(instance, route)
        score = -(LLM_dis - baseline) / baseline
        return score, instance_info

    def evaluate(self, heuristic: callable, ins_to_be_evaluated_id: Set | List | None = None, training_mode=True) -> \
    Optional[dict, list]:

        ins_to_be_evaluated_set = self.instance_set
        if not training_mode:
            ins_to_be_evaluated_set = self.ins_to_be_solve_set

        if not ins_to_be_evaluated_id:
            ins_to_be_evaluated_id = set(self.instance_set.keys())
            if not training_mode:
                ins_to_be_evaluated_id = set(self.ins_to_be_solve_set.keys())

        dis = []
        instance_infos = []
        instance_performance = {}
        n_ins = 0

        for ins_id in ins_to_be_evaluated_id:
            instance_infomation = ins_to_be_evaluated_set[ins_id]
            score, instance_info = self.evaluate_single(heuristic, instance_information=instance_infomation)
            dis.append(score)
            instance_infos.append(instance_info)
            n_ins += 1
            instance_performance[ins_id] = {
                'score': score
            }

            if n_ins == self.n_instance:
                break

        img_base64 = ''
        if self.multimodal_Flag:
            which_image = dis.index(min(dis))
            instance_info_seleted = instance_infos[which_image]

            try:
                # 显式传递所有必需参数
                fig = self.plot_clean_solution(instance_info_seleted['coordinates'],
                                               instance_info_seleted['route'],
                                               instance_info_seleted['demands'],
                                               instance_info_seleted['vehicle_capacity'])

                buffer = BytesIO()
                fig.savefig(buffer, format="png", bbox_inches='tight', dpi=80)  # dpi=80 减小体积
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

                plt.close(fig)  # 必须关闭 fig 以防内存泄漏
            except Exception as e:
                print(f"Plotting error: {e}")
                img_base64 = ""

        if self.whocall in ['eohs']:
            return (dis, instance_performance)
        elif self.whocall in ['dyca']:
            return {'all_ins_performance': instance_performance, 'list_performance': dis}
        elif self.whocall in ['eoh', 'funsearch']:
            return np.mean(dis)
        elif self.whocall in ['mmeoh', 'reevo']:
            return {'score': np.mean(dis), 'image': img_base64, 'observation': None,
                    'all_ins_performance': instance_performance, 'list_performance': dis}
        else:
            return np.mean(dis)

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        ins_to_be_evaluated_id = kwargs.get('ins_to_be_evaluated_id', None)
        training_mode = kwargs.get('training_mode', True)
        return self.evaluate(callable_func, ins_to_be_evaluated_id, training_mode)


if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    import io

    balanced_training_set = [
        # './balanced_trainingset/cvrp_clustered_cap30-60_n30-60_num20_train.pkl',
        './balanced_trainingset/cvrp_uniform_cap100-300_n500-800_num20_train.pkl',
        # './balanced_trainingset/cvrp_clustered_cap30-60_n80-120_num20_train.pkl',
        # './balanced_trainingset/cvrp_clustered_cap30-60_n180-200_num20_train.pkl',
        # './balanced_trainingset/cvrp_clustered_cap90-110_n80-120_num20_train.pkl',
        # './balanced_trainingset/cvrp_heavy_cap30-60_n30-60_num20_train.pkl',
        # './balanced_trainingset/cvrp_heavy_cap30-60_n80-120_num20_train.pkl',
        # './balanced_trainingset/cvrp_heavy_cap30-60_n180-200_num20_train.pkl',
        # './balanced_trainingset/cvrp_heavy_cap90-110_n80-120_num20_train.pkl',
        # './balanced_trainingset/cvrp_uniform_cap30-60_n30-60_num20_train.pkl',
        # './balanced_trainingset/cvrp_uniform_cap30-60_n80-120_num20_train.pkl',
        # './balanced_trainingset/cvrp_uniform_cap30-60_n180-200_num20_train.pkl',
        # './balanced_trainingset/cvrp_uniform_cap90-110_n80-120_num20_train.pkl',
    ]


    def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray,
                         demands: np.ndarray, distance_matrix: np.ndarray) -> int:
        """Design a novel algorithm to select the next node in each step.
        Args:
            current_node: ID of the current node.
            depot: ID of the depot.
            unvisited_nodes: Array of IDs of unvisited nodes.
            rest_capacity: rest capacity of vehicle
            demands: demands of nodes
            distance_matrix: Distance matrix of nodes.
        Return:
            ID of the next node to visit.
        """
        feasible_nodes = [node for node in unvisited_nodes if demands[node] <= rest_capacity]
        if not feasible_nodes:
            return depot
        scores = [demands[node] / distance_matrix[current_node][node] for node in feasible_nodes]
        return feasible_nodes[np.argmax(scores)]


    # def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    #     """Design a novel algorithm to select the next node in each step.
    #     Args:
    #         current_node: ID of the current node.
    #         depot: ID of the depot.
    #         unvisited_nodes: Array of IDs of unvisited nodes.
    #         rest_capacity: rest capacity of vehicle
    #         demands: demands of nodes
    #         distance_matrix: Distance matrix of nodes.
    #     Return:
    #         ID of the next node to visit.
    #     """
    #     best_score = -1
    #     next_node = -1

    #     for node in unvisited_nodes:
    #         demand = demands[node]
    #         distance = distance_matrix[current_node][node]

    #         if demand <= rest_capacity:
    #             score = demand / distance if distance > 0 else float('inf')  # Avoid division by zero
    #             if score > best_score:
    #                 best_score = score
    #                 next_node = node

    #     return next_node

    eval = CVRPSEvaluation(
        timeout_seconds=120,
        run_mode='Training',
        training_datasets=balanced_training_set,
        testing_datasets=balanced_training_set,
        whocall='mmeoh',
        objective_value=0)

    return_thing = eval.evaluate(select_next_node)

    # 2. 从返回的字典中提取数据
    avg_score = return_thing['score']
    img_b64 = return_thing['image']

    print(f"评估完成！平均得分 (Gap): {avg_score:.4f}")

    # 3. 可视化 Base64 图像
    if img_b64:
        print("正在解码并显示 Base64 图像...")
        # 解码 Base64
        img_data = base64.b64decode(img_b64)
        # 将字节流转换为图片对象
        img = Image.open(io.BytesIO(img_data))

        try:
            img.show()
        except Exception as e:
            print(f"无法直接弹出窗口: {e}")
    else:
        print("未获取到 Base64 图像数据。")

    print(img_b64)

    output_data = {
        "image_type": "png",  # 或者是 jpg，取决于您的生成源
        "image_base64": img_b64
    }

    filename = "eval_result_imgbase64.json"

    try:
        # 'w' 表示写入模式, encoding='utf-8' 确保字符编码正确
        with open(filename, 'w', encoding='utf-8') as f:
            # indent=4 让生成的 JSON 文件有缩进，方便人类阅读
            # ensure_ascii=False 防止中文或其他非ASCII字符变成乱码
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(f"✅ 成功！数据已保存到文件: {filename}")

    except Exception as e:
        print(f"❌ 保存 JSON 失败: {e}")

    # # --- 3. 提取一个实例进行可视化 ---
    # if len(eval.instance_set) > 0:
    #     # 取出第一个实例的 ID
    #     target_id = list(eval.instance_set.keys())[0]
    #     instance_info = eval.instance_set[target_id]
    #
    #     print(f"正在绘制实例 ID: {target_id}")
    #
    #     # 准备数据
    #     coords = instance_info['coordinates']  # 坐标
    #     dist_matrix = instance_info['distances']  # 距离矩阵
    #     demands = instance_info['demands']  # 需求
    #     capacity = instance_info['capacity']  # 车载容量
    #
    #     # 设置当前问题的规模 (route_construct 依赖这个变量)
    #     eval.problem_size = len(demands)
    #
    #     # --- 4. 生成路径 ---
    #     # 手动调用 route_construct 获得路径列表
    #     route = eval.route_construct(dist_matrix, demands, capacity, select_next_node)
    #
    #     print(f"生成的路径: {route}")
    #
    #     # --- 5. 画图 ---
    #     if route is not None:
    #         eval.plot_solution(coords, route, demands, capacity)
    #         eval.plot_clean_solution(coords, route, demands, capacity)
    #     else:
    #         print("生成路径失败（可能产生无效解）。")

    # res = eval.evaluate_program('', select_next_node)
    # print(res)
