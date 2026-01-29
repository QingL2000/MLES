import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import io
from io import BytesIO
import base64


def action_selection(observation):
    """
    一个简单的启发式策略来决定动作。

    参数:
    observation (np.ndarray): 环境返回的状态观测值，
                              (cart_pos, cart_vel, pole_angle, pole_vel)

    返回:
    int: 0 (向左推) 或 1 (向右推)
    """
    pole_angle = observation[2]
    pole_velocity = observation[3]

    # 这是一个简单的 PID 控制器（只用了 PD 部分）
    # 我们为角度和速度设置权重，然后将它们相加
    # 目标是让这个控制信号保持在 0 附近
    # 这些权重是经验值，你可以尝试调整它们
    angle_weight = 4.0
    velocity_weight = 2.0

    control_signal = (pole_angle * angle_weight) + (pole_velocity * velocity_weight)

    # 如果信号 > 0，意味着杆正在向右倾斜或移动，向右推
    # 如果信号 < 0，意味着杆正在向左倾斜或移动，向左推
    if control_signal > 0:
        action = 1  # 向右推
    else:
        action = 0  # 向左推

    return action

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

env = gym.make("CartPole-v1", render_mode="rgb_array")
# env = gym.make("CartPole-v1", render_mode="rgb_array")
#
# # 创建LunarLander-v3环境
# env1 = gym.make('LunarLander-v3', render_mode='rgb_array')
# env2 = gym.make('LunarLander-v3', render_mode='rgb_array')

# 重置环境
state, _ = env.reset(seed=0, options={"low": -0.1, "high": 0.1})
sum_reward = 0
done = False

step = 0

# 创建一个空白画布
canvas = np.ones((400, 600, 3), dtype=np.float32) * 255
# canvas2 = np.zeros((400, 600, 3), dtype=np.float32)

action = 0

state_pre, reward, done, t, info = env.step(action)
calculator = 0
while not done and step < 500:
    step += 1

    state_pre = state
    action = action_selection(state_pre)
    state, reward, done, t, info = env.step(action)
    sum_reward += reward

    print('type(state)', type(state))
    print(f"step: {step}, state: {state}, reward: {reward}, done: {done}, t: {t}, action: {action}")

    if calculator >= 20:
        # 获取当前帧的图像
        img = env.render()
        # print(img.shape)

        # 提取非白色部分
        mask = np.any(img != [255, 255, 255], axis=-1)

        # 计算动态透明度
        alpha = step / 500 # 假设最大步数为200，可以根据实际情况调整
        alpha = min(alpha, 1.0)  # 确保透明度不超过1

        # 将当前帧的非黑色部分叠加到画布上
        canvas[mask] = canvas[mask] * (1 - alpha) + img[mask] * alpha
        calculator = 0

    calculator += 1

# 获取canvas的base64编码
base64_canvas = image_to_base64(canvas, score=sum_reward)
# base64_canvas2 = image_to_base64(canvas2)

# 打印出base64编码
print(base64_canvas)
# print(base64_canvas2)

# 显示两个图像
display_base64_image(base64_canvas)
# display_base64_image(base64_canvas2)

# 关闭环境
env.close()
# env2.close()

save_base64_as_png(base64_canvas, "canvas_image.png")
# save_base64_as_png(base64_canvas2, "canvas_image2.png")