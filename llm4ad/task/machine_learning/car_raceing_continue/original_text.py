import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# env = gym.make('CarRacing-v3', render_mode='human')
env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=True)


def heuristic_action(observation):
    action = np.array([0.0, 0.0, 0.0])  # [steering, gas, brake]

    white_threshold = 200
    white_mask = (observation[:, :, 0] > white_threshold) & \
                 (observation[:, :, 1] > white_threshold) & \
                 (observation[:, :, 2] > white_threshold)

    white_indices = np.argwhere(white_mask)
    if len(white_indices) > 0:
        center_x = np.mean(white_indices[:, 1])
    else:
        center_x = observation.shape[1] // 2

    car_position = observation.shape[1] // 2
    offset = center_x - car_position


    if offset > 10:
        action[0] = 0.5
        action[1] = 0.0
        action[2] = 0.2
    elif offset < -10:
        action[0] = -0.5
        action[1] = 0.0
        action[2] = 0.2
    else:
        action[0] = 0.0
        action[1] = 0.8
        action[2] = 0.0


    white_density = np.sum(white_mask) / (observation.shape[0] * observation.shape[1])
    if white_density < 0.1:
        action[1] = 0.4
        action[2] = 0.3

    return action

def heuristic_action_4o(observation):
    action = np.array([0.0, 0.0, 0.0])
    white_threshold = 200
    white_mask = (observation[:, :, 0] > white_threshold) & \
                 (observation[:, :, 1] > white_threshold) & \
                 (observation[:, :, 2] > white_threshold)

    white_indices = np.argwhere(white_mask)
    center_x = np.mean(white_indices[:, 1]) if len(white_indices) > 0 else observation.shape[1] // 2
    car_position = observation.shape[1] // 2
    offset = center_x - car_position

    steering_angle = np.clip(offset / 100.0, -1.0, 1.0)
    action[0] = steering_angle

    if abs(offset) > 10:
        action[1] = 0.0
        action[2] = 0.2
    else:
        action[1] = 0.8
        action[2] = 0.0

    white_density = np.sum(white_mask) / (observation.shape[0] * observation.shape[1])
    if white_density < 0.1:
        action[1] = 0.4
        action[2] = 0.3

    return action


def display_observation(observation):

    plt.imshow(observation)
    plt.axis('off')
    plt.show()



observation, _ = env.reset(seed=42)

done = False
step = 0
sum_reward = 0
while not done:
    if step % 20 == 0:
        pre_observation = observation
    step += 1
    print(step)
    action = heuristic_action(observation)
    # action = heuristic_action_4o(observation)

    observation, reward, done, info, _ = env.step(action)
    sum_reward += reward
    print(step, sum_reward)

    # obs1_upper = observation[:83, :, :]
    obs1_upper = observation[:, :, :]

    if np.mean(pre_observation) == np.mean(observation):
        break
    if step > 200 and step % 30 == 0:
        display_observation(obs1_upper)

    env.render()

env.close()
display_observation(obs1_upper)
print(sum_reward)