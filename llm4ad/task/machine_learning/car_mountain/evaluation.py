


from __future__ import annotations

from typing import Any
import gym

from llm4ad.base import Evaluation
from llm4ad.task.machine_learning.car_mountain.template import template_program, task_description

__all__ = ['CarMountainEvaluation']


def evaluate(env: gym.Env, action_select: callable) -> float:
    """Evaluate heuristic function on car mountain problem."""

    observation, _ = env.reset()  # initialization
    action = 1  # initial action, stay static

    for i in range(env._max_episode_steps):
        action = action_select(observation[0], observation[1], action)
        observation, reward, done, truncated, info = env.step(action)

        if done:
            return -(i / env._max_episode_steps)  # succeed

        if truncated:
            return -(max(0.5 - observation[0], 0) + 1)  # failed


class CarMountainEvaluation(Evaluation):
    """Evaluator for car mountain problem."""

    def __init__(self, max_steps=500, timeout_seconds=20, **kwargs):
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

        self.env = None
        self.env = gym.make('MountainCar-v0')
        self.env._max_episode_steps = max_steps

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return evaluate(self.env, callable_func)
