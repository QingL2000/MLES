from __future__ import annotations

from typing import Optional, Tuple, List, Any
import gymnasium as gym
import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.machine_learning.car_raceing_continue.template import template_program, task_description

import traceback
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from io import BytesIO
import base64
import copy
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

__all__ = ['RacingCarEvaluation']


# def evaluate(env: gym.Env, action_select: callable) -> float | None:


class RacingCarEvaluation(Evaluation):
    """Evaluator for Car Racing problem."""

    def __init__(self, whocall='Eoh', max_steps=1200, timeout_seconds=180, **kwargs):
        """
            Args:
                - 'max_steps' (int): Maximum number of steps allowed per episode in the MountainCar-v0 environment (default is 500).
                - '**kwargs' (dict): Additional keyword arguments passed to the parent class initializer.

            Attributes:
                - 'env' (gym.Env): The MountainCar-v0 environment with a modified maximum episode length.
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self.env_name = "CarRacing-v3"
        self.env_max_episode_steps = max_steps
        self.whocall = whocall
        self.env_mode = kwargs.get("env_mode", 'rgb_array')

    def evaluate(self, action_select: callable, env_seeds=(40, 1231, 516, 413), skip_frame=1) -> Optional[dict]:
        try:
            total_rewards = []
            image64s = []
            num_episodes = len(env_seeds)
            episodes_recorder = {}
            for i in range(num_episodes):
                each_evaluate_result = self.evaluate_single(action_select, env_seed=env_seeds[i], skip_frame=skip_frame)
                if each_evaluate_result is not None:
                    infos = each_evaluate_result[0]
                    total_rewards.append(infos['track_coverage'])
                    image64s.append(each_evaluate_result[1])
                    episodes_recorder[f'{i}'] = infos

            mean_reward = np.mean(total_rewards)

            which_image = total_rewards.index(min(total_rewards))
            chosen_image_base64 = image64s[which_image]

            nws = mean_reward

            test_result = {
                'Mean Reward': mean_reward,
                'NWS': nws
            }

            if self.whocall == 'mmeoh':
                return {'score': nws, 'image': chosen_image_base64, 'Test result': episodes_recorder,
                        'observation': None,
                        'Test result for test': test_result}
            else:
                return nws
        except Exception as e:
            print(e)
            traceback.print_exc()
            return None

    def merge_evaluate(self, action_selects: List[callable], env_seeds=(40, 1231, 516, 413), skip_frame=1) -> Optional[
        dict]:
        try:
            total_rewards = []
            image64s = []
            num_episodes = len(env_seeds)
            episodes_recorder = {}
            for i in range(num_episodes):
                each_evaluate_result = self.evaluate_single_merge(action_selects, env_seed=env_seeds[i],
                                                                  skip_frame=skip_frame)
                if each_evaluate_result is not None:
                    infos = each_evaluate_result[0]
                    total_rewards.append(infos['track_coverage'])
                    image64s.append(each_evaluate_result[1])
                    episodes_recorder[f'{i}'] = infos

            mean_reward = np.mean(total_rewards)

            which_image = total_rewards.index(min(total_rewards))
            chosen_image_base64 = image64s[which_image]

            nws = mean_reward

            test_result = {
                'Mean Reward': mean_reward,
                'NWS': nws
            }

            if self.whocall == 'mmeoh':
                return {'score': nws, 'image': chosen_image_base64, 'Test result': episodes_recorder,
                        'observation': None,
                        'Test result for test': test_result}
            else:
                return nws
        except Exception as e:
            print(e)
            traceback.print_exc()
            return None

    def evaluate_single(self, action_select: callable, env_seed=42, skip_frame=1):
        """Evaluate heuristic function on racing car problem."""
        env = gym.make(self.env_name, render_mode=self.env_mode, domain_randomize=False, continuous=True)  # 'rgb_array'
        observation, _ = env.reset(seed=env_seed)  # initialization

        action = np.array([0.0, 1.0, 0.0])  # initial action
        episode_reward = 0
        episode_max_reward = 0

        trajectory = []
        car_angles = []
        view_rectangles = []
        done = False

        view_length = 46.0
        view_width = 38.0
        view_offset = 14.0

        pre_observation = copy.deepcopy(observation)
        observation, reward, done, truncated, info = env.step(action)
        episode_reward += reward

        step = 0
        while not done and step < self.env_max_episode_steps:
            car_velocity = env.unwrapped.car.hull.linearVelocity
            speed = np.sqrt(car_velocity[0] ** 2 + car_velocity[1] ** 2)
            action = action_select(observation,
                                   speed,
                                   action,
                                   pre_observation)
            pre_observation = copy.deepcopy(observation)

            for _ in range(skip_frame):
                observation, reward, done, truncated, info = env.step(action)
                step += 1

                car_pos = env.unwrapped.car.hull.position
                car_angle = env.unwrapped.car.hull.angle

                trajectory.append((car_pos.x, car_pos.y))
                car_angles.append(car_angle)

                corrected_angle = car_angle + np.pi / 2

                view_center_x = car_pos.x + np.cos(corrected_angle) * view_offset
                view_center_y = car_pos.y + np.sin(corrected_angle) * view_offset

                view_rectangles.append((view_center_x, view_center_y, corrected_angle, view_width, view_length))

                episode_reward += reward
                episode_max_reward = max(episode_max_reward, episode_reward)

        plt.figure(figsize=(9, 8))
        green_color = '#62f972'
        plt.gca().set_facecolor(green_color)

        for polygon in env.unwrapped.road_poly:
            vertices = polygon[0]
            color = polygon[1]

            if hasattr(color, '__iter__') and not isinstance(color, tuple):
                color = tuple(color)

            fill_color = '#666666'

            if isinstance(color, tuple) and len(color) == 3:
                r = max(0, min(255, int(round(color[0]))))
                g = max(0, min(255, int(round(color[1]))))
                b = max(0, min(255, int(round(color[2]))))

                fill_color = "#{:02X}{:02X}{:02X}".format(r, g, b)

            x_coords = [v[0] for v in vertices] + [vertices[0][0]]
            y_coords = [v[1] for v in vertices] + [vertices[0][1]]

            plt.fill(x_coords, y_coords, color=fill_color, alpha=1.0)

        view_color = '#8000FF'
        arrow_interval = 40
        for idy, rect in enumerate(view_rectangles):
            if idy == 0 or idy == len(view_rectangles) - 1 or idy % arrow_interval == 0:
                center_x, center_y, angle, length, width = rect

                rect_patch = patches.Rectangle(
                    (-length / 2, -width / 2),
                    length,
                    width,
                    linewidth=0,
                    edgecolor='none',
                    facecolor=view_color,
                    alpha=0.1
                )

                t = Affine2D().rotate(angle).translate(center_x, center_y) + plt.gca().transData
                rect_patch.set_transform(t)
                plt.gca().add_patch(rect_patch)

        arrow_color = '#FF6A00'

        if trajectory:
            trajectory = np.array(trajectory)
            plt.plot(trajectory[:, 0], trajectory[:, 1], '-', color='#FFD700', linewidth=1, label='Trajectory')
            # plt.scatter(trajectory[0, 0], trajectory[0, 1], c='#1E90FF', s=100, label='Start Point')
            # plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='#FF00FF', s=100, label='End Point')

            for i in range(len(trajectory)):

                if i == 0 or i == len(trajectory) - 1 or i % arrow_interval == 0:
                    x, y = trajectory[i, 0], trajectory[i, 1]
                    angle = car_angles[i] + np.pi / 2
                    dx = np.cos(angle) * 3
                    dy = np.sin(angle) * 5

                    arrow_start_x = x - dx * 0.3
                    arrow_start_y = y - dy * 0.3

                    plt.arrow(arrow_start_x, arrow_start_y, dx, dy,
                              head_width=3, head_length=4, fc=arrow_color, ec=arrow_color)

        grass_patch = patches.Patch(color=green_color, label='Off-Track Area (Grass)')
        track_patch = patches.Patch(color='#666666', label='Track')
        border_patch = patches.Patch(color='red', label='Curbing (red-white pattern at sharp turns)')
        view_patch = patches.Patch(color=view_color, alpha=0.1, label="Agent's Dynamic Visual Field")

        handles, labels = plt.gca().get_legend_handles_labels()

        custom_handles = [grass_patch, track_patch, border_patch, view_patch]
        all_handles = custom_handles + handles

        seen_labels = set()
        unique_handles = []
        for handle in all_handles:
            label = handle.get_label()
            if label not in seen_labels:
                seen_labels.add(label)
                unique_handles.append(handle)

        track_coverage = env.unwrapped.tile_visited_count / len(env.unwrapped.track) * 100

        plt.title(
            f"Track with Car Trajectory and Corresponding Dynamic View Areas\n"
            f"Track Completion Rate: {track_coverage:.1f} %")

        plt.axis('equal')
        plt.legend(handles=unique_handles)

        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches='tight')
        buffer.seek(0)

        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        plt.close()
        env.close()

        infos = {'done': done,
                 'truncated': truncated,
                 'episode_reward': episode_reward,
                 'track_coverage': track_coverage,
                 'episode_max_reward': episode_max_reward}
        return infos, img_base64

    def evaluate_single_merge(self, action_selects: List[callable], env_seed=42, skip_frame=1):
        """Evaluate heuristic function on racing car problem."""
        env = gym.make(self.env_name, render_mode=self.env_mode, domain_randomize=False, continuous=True)  # 'rgb_array'
        observation, _ = env.reset(seed=env_seed)  # initialization

        action = np.array([0.0, 1.0, 0.0])  # initial action
        episode_reward = 0
        episode_max_reward = 0

        trajectory = []
        car_angles = []
        view_rectangles = []
        done = False

        view_length = 46.0
        view_width = 38.0
        view_offset = 14.0

        pre_observation = copy.deepcopy(observation)
        observation, reward, done, truncated, info = env.step(action)
        episode_reward += reward

        step = 0
        while not done and step < self.env_max_episode_steps:
            car_velocity = env.unwrapped.car.hull.linearVelocity
            speed = np.sqrt(car_velocity[0] ** 2 + car_velocity[1] ** 2)
            # Initialize list to store actions from all policies
            all_actions = []

            for policy in action_selects:
                try:
                    policy_action = policy(observation, speed, action, pre_observation)
                    # Validate the action shape and values
                    if not isinstance(policy_action, np.ndarray) or policy_action.shape != (3,):
                        raise ValueError("Invalid action shape")
                    all_actions.append(policy_action)
                except Exception as e:
                    print(f"Policy failed with error: {str(e)}. Using default action.")
                    default_action = np.array([0.0, 0.5, 0.5])  # [steering, gas, brake]
                    all_actions.append(default_action)

            if all_actions:
                # Convert to numpy array for vector operations
                all_actions = np.array(all_actions)

                # Calculate mean steering (range -1 to 1)
                steering = np.mean(all_actions[:, 0])
                steering = np.clip(steering, -1, 1)

                # Calculate mean gas (range 0 to 1)
                gas = np.mean(all_actions[:, 1])
                gas = np.clip(gas, 0, 1)

                # Calculate mean brake (range 0 to 1)
                brake = np.mean(all_actions[:, 2])
                brake = np.clip(brake, 0, 1)

                action = np.array([steering, gas, brake])

            pre_observation = copy.deepcopy(observation)

            for _ in range(skip_frame):
                observation, reward, done, truncated, info = env.step(action)
                step += 1

                car_pos = env.unwrapped.car.hull.position
                car_angle = env.unwrapped.car.hull.angle

                trajectory.append((car_pos.x, car_pos.y))
                car_angles.append(car_angle)

                corrected_angle = car_angle + np.pi / 2

                view_center_x = car_pos.x + np.cos(corrected_angle) * view_offset
                view_center_y = car_pos.y + np.sin(corrected_angle) * view_offset

                view_rectangles.append((view_center_x, view_center_y, corrected_angle, view_width, view_length))

                episode_reward += reward
                episode_max_reward = max(episode_max_reward, episode_reward)

        plt.figure(figsize=(9, 8))
        green_color = '#62f972'
        plt.gca().set_facecolor(green_color)

        for polygon in env.unwrapped.road_poly:
            vertices = polygon[0]
            color = polygon[1]

            if hasattr(color, '__iter__') and not isinstance(color, tuple):
                color = tuple(color)

            fill_color = '#666666'

            if isinstance(color, tuple) and len(color) == 3:
                r = max(0, min(255, int(round(color[0]))))
                g = max(0, min(255, int(round(color[1]))))
                b = max(0, min(255, int(round(color[2]))))

                fill_color = "#{:02X}{:02X}{:02X}".format(r, g, b)

            x_coords = [v[0] for v in vertices] + [vertices[0][0]]
            y_coords = [v[1] for v in vertices] + [vertices[0][1]]

            plt.fill(x_coords, y_coords, color=fill_color, alpha=1.0)

        view_color = '#8000FF'
        arrow_interval = 40

        for idy, rect in enumerate(view_rectangles):
            if idy == 0 or idy == len(view_rectangles) - 1 or idy % arrow_interval == 0:
                center_x, center_y, angle, length, width = rect

                rect_patch = patches.Rectangle(
                    (-length / 2, -width / 2),
                    length,
                    width,
                    linewidth=0,
                    edgecolor='none',
                    facecolor=view_color,
                    alpha=0.1
                )

                t = Affine2D().rotate(angle).translate(center_x, center_y) + plt.gca().transData
                rect_patch.set_transform(t)
                plt.gca().add_patch(rect_patch)

        arrow_color = '#FF6A00'

        if trajectory:
            trajectory = np.array(trajectory)
            plt.plot(trajectory[:, 0], trajectory[:, 1], '-', color='#FFD700', linewidth=1, label='Trajectory')
            # plt.scatter(trajectory[0, 0], trajectory[0, 1], c='#1E90FF', s=100, label='Start Point')
            # plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='#FF00FF', s=100, label='End Point')

            for i in range(len(trajectory)):

                if i == 0 or i == len(trajectory) - 1 or i % arrow_interval == 0:
                    x, y = trajectory[i, 0], trajectory[i, 1]
                    angle = car_angles[i] + np.pi / 2
                    dx = np.cos(angle) * 3
                    dy = np.sin(angle) * 5

                    arrow_start_x = x - dx * 0.3
                    arrow_start_y = y - dy * 0.3

                    plt.arrow(arrow_start_x, arrow_start_y, dx, dy,
                              head_width=3, head_length=4, fc=arrow_color, ec=arrow_color)

        grass_patch = patches.Patch(color=green_color, label='Off-Track Area (Grass)')
        track_patch = patches.Patch(color='#666666', label='Track')
        border_patch = patches.Patch(color='red', label='Curbing (red-white pattern at sharp turns)')
        view_patch = patches.Patch(color=view_color, alpha=0.1, label="Agent's Dynamic Visual Field")

        handles, labels = plt.gca().get_legend_handles_labels()

        custom_handles = [grass_patch, track_patch, border_patch, view_patch]
        all_handles = custom_handles + handles

        seen_labels = set()
        unique_handles = []
        for handle in all_handles:
            label = handle.get_label()
            if label not in seen_labels:
                seen_labels.add(label)
                unique_handles.append(handle)

        track_coverage = env.unwrapped.tile_visited_count / len(env.unwrapped.track) * 100

        plt.title(
            f"Track with Car Trajectory and Corresponding Dynamic View Areas\n"
            f"Track Completion Rate: {track_coverage:.1f} %")

        plt.axis('equal')
        plt.legend(handles=unique_handles)

        # 2. 保存到缓冲区
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches='tight')
        buffer.seek(0)

        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        plt.close()
        env.close()

        infos = {'done': done,
                 'truncated': truncated,
                 'episode_reward': episode_reward,
                 'track_coverage': track_coverage,
                 'episode_max_reward': episode_max_reward}
        return infos, img_base64

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return self.evaluate(callable_func)
