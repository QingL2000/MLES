import random
import gymnasium as gym
import numpy as np
import random

# original
def choose_action(s: np.ndarray, last_action: int, s_pre: np.ndarray) -> int:
    """
    Selects an action for the Lunar Lander to achieve a safe landing at the target location (0, 0).

    Args:
        s (ndarray): The current state of the lander. Elements:
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

        s_pre (ndarray): The state of the lander *before* the previous action was executed. Same structure as `s`.

    Returns:
        int: The chosen action. One of:
            0 - do nothing
            1 - fire left orientation engine
            2 - fire main (upward) engine
            3 - fire right orientation engine
    """
    # Compute desired angle: try to steer toward center, with damping from horizontal velocity
    angle_targ = s[0] * 0.5 + s[2] * 1.0
    angle_targ = np.clip(angle_targ, -0.4, 0.4)  # limit angle within safe bounds (~±22 degrees)

    # Compute desired hover height based on distance from center
    hover_targ = 0.55 * np.abs(s[0])

    # Determine adjustments needed for angle and vertical position
    angle_todo = (angle_targ - s[4]) * 0.5 - s[5] * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - s[3] * 0.5

    # If legs are in contact with ground, focus only on reducing vertical speed
    if s[6] or s[7]:
        angle_todo = 0
        hover_todo = -s[3] * 0.5

    # Action decision logic
    a = 0  # default: do nothing
    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
        a = 2  # fire main engine to control descent
    elif angle_todo < -0.05:
        a = 3  # fire right engine to rotate left
    elif angle_todo > 0.05:
        a = 1  # fire left engine to rotate right

    return a

def choose_action(s: list, last_action: int, s_pre: list) -> int:
    """
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

        s_pre (list or np.ndarray): The state of the lander *before* the last_action was executed.

    Returns:
        int: The chosen action for the next step. One of:
            0 - do nothing
            1 - fire left orientation engine
            2 - fire main (upward) engine
            3 - fire right orientation engine
    """
    # Target angle based on horizontal offset and velocity
    angle_target = (s[0] * 0.4) + (s[2] * 0.6)
    angle_target = np.clip(angle_target, -0.1, 0.1)

    # Hover target based on horizontal offset
    hover_target = np.fabs(s[0]) * 0.5

    # Adjustments
    angle_adjustment = (angle_target - s[4]) * 0.6 - (s[5] * 0.8)
    hover_adjustment = (hover_target - s[1]) * 0.5 - (s[3] * 0.5)

    # If any leg has contact, modify adjustments
    if s[6] or s[7]:
        angle_adjustment = 0
        hover_adjustment = -s[3] * 0.9

    action = 0  # Default: do nothing

    if s[1] < 1.0:  # Low altitude
        if abs(angle_adjustment) > 0.05:
            action = 1 if angle_adjustment > 0 else 3
        if hover_adjustment > 0.1:
            action = 2

    elif s[1] < 2.5:  # Medium altitude
        if hover_adjustment > 0.1:
            action = 2
        elif abs(angle_adjustment) > 0.05:
            action = 1 if angle_adjustment > 0 else 3

    else:  # High altitude
        if abs(angle_target) > 0.05:
            action = 1 if angle_adjustment > 0 else 3
        if hover_adjustment > 0.1:
            action = 2

    return action

# 创建LunarLander-v2环境
env = gym.make('LunarLander-v3', render_mode='human')


seeds = [42, 520, 1231, 114, 886]
# 重置环境
state, _ = env.reset(seed = seeds[4])

action = 0
state_pre, reward, done, t, info = env.step(action)

done = False
sum_reward = 0

step = 0
while not done and step < 200:
    step += 1

    action = choose_action(state, 0, state_pre)
    state_pre = state

    # 环境采取动作并返回新的状态、奖励等
    state, reward, done, t, info = env.step(action)

    print(f"step: {step}, state: {state}, reward: {reward}, done: {done}, t: {t}, action: {action}")

    # 渲染环境
    env.render()
    sum_reward += reward

# 关闭环境
env.close()
print('reward总和', sum_reward)
