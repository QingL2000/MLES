import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import io
from io import BytesIO
import base64


# def choose_action(s: np.ndarray, last_action: int, s_pre: np.ndarray) -> int:
#     """
#     Selects an action for the Lunar Lander to achieve a safe landing at the target location (0, 0).
#
#     Args:
#         s (ndarray): The current state of the lander. Elements:
#             s[0] - horizontal position (x)
#             s[1] - vertical position (y)
#             s[2] - horizontal velocity (v_x)
#             s[3] - vertical velocity (v_y)
#             s[4] - angle (radians)
#             s[5] - angular velocity
#             s[6] - 1 if the first leg is in contact with the ground, else 0
#             s[7] - 1 if the second leg is in contact with the ground, else 0
#
#         last_action (int): The action taken in the previous step. One of:
#             0 - do nothing
#             1 - fire left orientation engine
#             2 - fire main (upward) engine
#             3 - fire right orientation engine
#
#         s_pre (ndarray): The state of the lander *before* the previous action was executed. Same structure as `s`.
#
#     Returns:
#         int: The chosen action. One of:
#             0 - do nothing
#             1 - fire left orientation engine
#             2 - fire main (upward) engine
#             3 - fire right orientation engine
#     """
#     # Compute desired angle: try to steer toward center, with damping from horizontal velocity
#     angle_targ = s[0] * 0.5 + s[2] * 1.0
#     angle_targ = np.clip(angle_targ, -0.4, 0.4)  # limit angle within safe bounds (~±22 degrees)
#
#     # Compute desired hover height based on distance from center
#     hover_targ = 0.55 * np.abs(s[0])
#
#     # Determine adjustments needed for angle and vertical position
#     angle_todo = (angle_targ - s[4]) * 0.5 - s[5] * 1.0
#     hover_todo = (hover_targ - s[1]) * 0.5 - s[3] * 0.5
#
#     # If legs are in contact with ground, focus only on reducing vertical speed
#     if s[6] or s[7]:
#         angle_todo = 0
#         hover_todo = -s[3] * 0.5
#
#     # Action decision logic
#     a = 0  # default: do nothing
#     if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
#         a = 2  # fire main engine to control descent
#     elif angle_todo < -0.05:
#         a = 3  # fire right engine to rotate left
#     elif angle_todo > 0.05:
#         a = 1  # fire left engine to rotate right
#
#     return a
#
# import numpy as np

# def choose_action(s: list, last_action: int, s_pre: list) -> int:
#     """
#     MMEOH
#     Selects an action for the Lunar Lander to achieve a safe landing at the target location (0, 0).
#
#     Args:
#         s (list or np.ndarray): The current state of the lander. Elements:
#             s[0] - horizontal position (x)
#             s[1] - vertical position (y)
#             s[2] - horizontal velocity (v_x)
#             s[3] - vertical velocity (v_y)
#             s[4] - angle (radians)
#             s[5] - angular velocity
#             s[6] - 1 if the first leg is in contact with the ground, else 0
#             s[7] - 1 if the second leg is in contact with the ground, else 0
#
#         last_action (int): The action taken in the previous step. One of:
#             0 - do nothing
#             1 - fire left orientation engine
#             2 - fire main (upward) engine
#             3 - fire right orientation engine
#
#         s_pre (list or np.ndarray): The state of the lander *before* the last action was executed. Elements:
#             s_pre[0] to s_pre[7]: Same structure as `s`
#
#     Returns:
#         int: The chosen action for the next step. One of:
#             0 - do nothing
#             1 - fire left orientation engine
#             2 - fire main (upward) engine
#             3 - fire right orientation engine
#     """
#     # Target angle and hover values based on position and velocity
#     angle_targ = s[0] * 0.2 + s[2] * 0.5
#     angle_targ = np.clip(angle_targ, -0.3, 0.3)
#
#     hover_targ = max(0.0, s[1]) * 0.7
#
#     # Compute required control changes
#     angle_todo = (angle_targ - s[4]) * 1.5 - s[5]
#     hover_todo = (hover_targ - s[1]) * 3.0 - s[3] * 1.2
#
#     # Adjust control parameters based on altitude
#     if s[1] > 6.0:
#         hover_todo *= 0.01
#         angle_todo *= 0.05
#     elif s[1] > 4.0:
#         hover_todo *= 0.1
#         angle_todo *= 0.1
#     elif s[1] > 2.0:
#         hover_todo *= 0.25
#         angle_todo *= 0.25
#     else:
#         hover_todo *= 0.4
#         angle_todo *= 0.4
#
#     # Evaluate historical states for predictive adjustment
#     historical_stability = np.mean([abs(s_pre[3]), abs(s_pre[5])])
#     if historical_stability < 0.1:
#         angle_todo *= 0.7
#         hover_todo *= 0.6
#
#     # Safe landing detection
#     safe_landing = abs(s[3]) < 0.03 and abs(s[5]) < 0.03 and s[6] and s[7]
#
#     # If touching ground
#     if s[6] or s[7]:
#         if safe_landing:
#             return 0
#         hover_todo = -s[3] * 5.0  # Apply upward force if landing is not stable
#
#     if safe_landing:
#         return 0
#
#     # Decision logic
#     a = 0
#     if hover_todo > np.abs(angle_todo) and hover_todo > 0.1:
#         a = 2
#     elif angle_todo < -0.05:
#         a = 3
#     elif angle_todo > 0.05:
#         a = 1
#
#     # Smooth out action changes
#     if last_action == a:
#         return a
#     elif np.abs(angle_todo) > 0.2 or np.abs(s[5]) > 0.15:
#         return a
#
#     return a

# def choose_action(s: list, last_action: int, s_pre: list) -> int:
#     """
#     Eoh
#     Selects an action for the Lunar Lander to achieve a safe landing at the target location (0, 0).
#
#     Args:
#         s (list or np.ndarray): The current state of the lander. Elements:
#             s[0] - horizontal position (x)
#             s[1] - vertical position (y)
#             s[2] - horizontal velocity (v_x)
#             s[3] - vertical velocity (v_y)
#             s[4] - angle (radians)
#             s[5] - angular velocity
#             s[6] - 1 if the first leg is in contact with the ground, else 0
#             s[7] - 1 if the second leg is in contact with the ground, else 0
#
#         last_action (int): The action taken in the previous step. One of:
#             0 - do nothing
#             1 - fire left orientation engine
#             2 - fire main (upward) engine
#             3 - fire right orientation engine
#
#         s_pre (list or np.ndarray): The state of the lander *before* the last action was executed.
#
#     Returns:
#         int: The chosen action for the next step. One of:
#             0 - do nothing
#             1 - fire left orientation engine
#             2 - fire main (upward) engine
#             3 - fire right orientation engine
#     """
#     # Target angle calculation based on position and velocity
#     angle_targ = np.clip(s[0] * 0.4 + s[2] * 0.6, -0.5, 0.5)
#
#     # Target hover height for descent management
#     hover_targ = 0.25 * s[1]
#
#     # Adjustments to correct current state
#     angle_todo = (angle_targ - s[4]) * 0.8 - s[5] * 1.0
#     hover_todo = (hover_targ - s[1]) * 0.5 - s[3] * 0.5
#
#     # If both legs are in contact with the ground, stabilize
#     if s[6] and s[7]:
#         angle_todo = 0
#         hover_todo = -s[3] * 0.2  # Gently slow down vertical velocity
#
#     # Determine the appropriate action
#     a = 0  # Default: do nothing
#     if hover_todo > np.abs(angle_todo) and hover_todo > 0.1:
#         a = 2  # Fire main engine
#     elif angle_todo < -0.1:
#         a = 3  # Fire right engine (rotate left)
#     elif angle_todo > 0.1:
#         a = 1  # Fire left engine (rotate right)
#
#     return a

def choose_action(s: list, last_action: int, s_pre: list) -> int:
    """
    Eoh+M_m1+M_m2
    Selects an action for the Lunar Lander to achieve a safe landing at the target location (0, 0).

    Args:
        s (list or np.ndarray): The current state of the lander. Elements:
            s[0] - horizontal position (x)
            s[1] - vertical position (y)
            s[2] - horizontal velocity (v_x)
            s[3] - vertical velocity (v_y)
            s[4] - angle (radians)
            s[5] - angular velocity
            s[6] - 1 if the first leg is in contact with the ground, else 0
            s[7] - 1 if the second leg is in contact with the ground, else 0

        last_action (int): The action taken in the previous step. One of:
            0 - do nothing
            1 - fire left orientation engine
            2 - fire main (upward) engine
            3 - fire right orientation engine

        s_pre (list or np.ndarray): The state of the lander *before* the last action was executed. Elements:
            s_pre[0] to s_pre[7]: Same structure as `s`

    Returns:
        int: The chosen action for the next step. One of:
            0 - do nothing
            1 - fire left orientation engine
            2 - fire main (upward) engine
            3 - fire right orientation engine
    """
    # Compute target angle and vertical position
    target_angles = np.clip(-0.5 * s[0] + 0.7 * s[2], -0.25, 0.25)
    target_vertical = max(0.2, 0.5 * abs(s[0]) + 0.5 * abs(s[3]))

    # Calculate control differences
    angle_diff = (target_angles - s[4]) * 0.4 - s[5] * 0.5
    vertical_diff = (target_vertical - s[1]) * 1.2 - s[3] * 0.7

    # Modify urgency if falling too fast or recent change is too sudden
    if s[3] < -2.0 or (s_pre[3] - s[3]) > 0.5:
        vertical_diff *= 2.0

    # Adjust control when legs are on the ground
    if s[6] and s[7]:  # Both legs in contact
        angle_diff *= 0.1
        vertical_diff = max(-s[3] * 0.4, 0)

    # Decision logic based on control diffs
    if vertical_diff > 0.3:
        return 2  # Fire main engine
    elif angle_diff < -0.03:
        return 3  # Fire right engine
    elif angle_diff > 0.03:
        return 1  # Fire left engine

    return 0  # Do nothing if stable



def image_to_base64(image, score='inf'):
    # 将图像保存为内存文件
    img_bytes = io.BytesIO()
    plt.imshow(image.astype(np.uint8))
    plt.title(f"Total reward:{score}")
    plt.axis('off')
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)

    # 对图像进行base64编码
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    return img_base64


# 用于解码并可视化base64图像
def display_base64_image(base64_str):
    # 解码base64字符串
    img_data = base64.b64decode(base64_str)
    img = BytesIO(img_data)

    # 显示图像
    img = plt.imread(img)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def save_base64_as_png(base64_str, file_name):
    # 解码base64字符串
    img_data = base64.b64decode(base64_str)

    # 将解码后的数据保存为PNG文件
    with open(file_name, 'wb') as f:
        f.write(img_data)
    print(f"图像已保存为 {file_name}")



# # 创建LunarLander-v3环境
# env1 = gym.make('LunarLander-v3', render_mode='rgb_array')
# env2 = gym.make('LunarLander-v3', render_mode='rgb_array')
#
# # 重置环境
# state, _ = env1.reset()
#
# done = False
#
# step = 0
#
# # 创建一个空白画布
# canvas = np.zeros((400, 600, 3), dtype=np.float32)
#
# action = 0
# while not done:
#     step += 1
#     action = choose_action(state, 0, action)
#
#     state, reward, done, t, info = env1.step(action)
#
#     print(f"step: {step}, state: {state}, reward: {reward}, done: {done}, t: {t}, action: {action}")
#     _, _ = env2.reset()
#     # 获取当前帧的图像
#     img = env1.render()
#
#     # 提取非黑色部分
#     mask = np.any(img != [0, 0, 0], axis=-1)
#
#     # 计算动态透明度
#     alpha = step / 100 # 假设最大步数为200，可以根据实际情况调整
#     alpha = min(alpha, 1.0)  # 确保透明度不超过1
#
#     # 将当前帧的非黑色部分叠加到画布上
#     canvas[mask] = canvas[mask] * (1 - alpha) + img[mask] * alpha
#
# # 使用matplotlib显示并保存图像
# plt.imshow(canvas.astype(np.uint8))
# plt.axis('off')
# plt.savefig('lander_trajectory.png')
# plt.show()
#
# # 关闭环境
# env1.close()
# env2.close()


# 创建LunarLander-v3环境
env1 = gym.make('LunarLander-v3', render_mode='rgb_array')
env2 = gym.make('LunarLander-v3', render_mode='rgb_array')

# 重置环境
state, _ = env1.reset(seed=1232)
sum_reward = 0
done = False

step = 0

# 创建一个空白画布
canvas = np.zeros((400, 600, 3), dtype=np.float32)
canvas2 = np.zeros((400, 600, 3), dtype=np.float32)

action = 0

state_pre, reward, done, t, info = env1.step(action)
calculator = 0
while not done and step < 200:
    step += 1

    action = choose_action(state, 0, state_pre)
    state_pre = state
    state, reward, done, t, info = env1.step(action)
    sum_reward += reward

    print('type(state)', type(state))
    print(f"step: {step}, state: {state}, reward: {reward}, done: {done}, t: {t}, action: {action}")
    _, _ = env2.reset()

    if calculator >= 8:
        # 获取当前帧的图像
        img = env1.render()
        img2 = env2.render()
        print(img.shape)

        # 提取非黑色部分
        mask = np.any(img != [0, 0, 0], axis=-1)
        mask2 = np.any(img2 != [0, 0, 0], axis=-1)

        # 计算动态透明度
        alpha = step / 100 # 假设最大步数为200，可以根据实际情况调整
        alpha = min(alpha, 1.0)  # 确保透明度不超过1



        # 将当前帧的非黑色部分叠加到画布上
        canvas[mask] = canvas[mask] * (1 - alpha) + img[mask] * alpha
        canvas2[mask2] = canvas2[mask2] * (1 - alpha) + img[mask2] * alpha
        calculator = 0

    calculator += 1

# 获取canvas的base64编码
base64_canvas = image_to_base64(canvas, score=sum_reward)
base64_canvas2 = image_to_base64(canvas2)

# 打印出base64编码
print(base64_canvas)
print(base64_canvas2)

# 显示两个图像
display_base64_image(base64_canvas)
display_base64_image(base64_canvas2)

# 关闭环境
env1.close()
env2.close()

save_base64_as_png(base64_canvas, "canvas_image.png")
save_base64_as_png(base64_canvas2, "canvas_image2.png")