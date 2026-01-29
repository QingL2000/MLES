template_program = '''
import numpy as np

def choose_action(s: list, last_action: int, s_pre: list) -> int:
    """
    Selects an action for the CartPole to keep it balanced.

    Args:
        s (list or np.ndarray): The current state of the CartPole. Elements:
            s[0] - Cart Position
            s[1] - Cart Velocity
            s[2] - Pole Angle (radians)
            s[3] - Pole Angular Velocity

        last_action (int): The action taken in the previous step. One of:
            0 - push cart to the left
            1 - push cart to the right

        s_pre (list or np.ndarray): The state of the CartPole *before* the last action was executed. Elements:
            s_pre[0] - Cart Position (previous state)
            s_pre[1] - Cart Velocity (previous state)
            s_pre[2] - Pole Angle (previous state)
            s_pre[3] - Pole Angular Velocity (previous state)

    Returns:
        int: The chosen action for the next step. One of:
            0 - push cart to the left
            1 - push cart to the right
    """
    pole_angle = s[2]
    pole_velocity = s[3]

    angle_weight = 4.0
    velocity_weight = 2.0

    control_signal = (pole_angle * angle_weight) + (pole_velocity * velocity_weight)

    if control_signal > 0:
        action = 1  
    else:
        action = 0  
    return action
'''

task_description = (
    "Implement a novel heuristic strategy function that guides the agent (the cart) in selecting actions step-by-step "
    "to keep the pole balanced. At each step, an appropriate action should be chosen based on the agent's "
    "current state and previous state, with the objective of maximizing the number of steps the pole remains balanced. "
    "A 'successful balance' is defined as keeping the pole upright without falling over within the environment's "
    "predefined limits (pole angle within approx. ±0.209 rad (±12 degrees) and cart position within approx. ±2.4 units)."
)

non_image_representation_explanation = ("The execution result is a two-dimensional list where each element is a 4-dimensional "
                                        "vector that records the state of the CartPole system during its balancing process. "
                                        "Each vector contains: the cart's position, its linear velocity, the pole's angle, "
                                        "and its angular velocity. The units of the state are as follows: 'Cart Position': (meters), "
                                        "'Cart Velocity': (meters/second), 'Pole Angle': (radians), 'Pole Angular Velocity': (radians/second).")
