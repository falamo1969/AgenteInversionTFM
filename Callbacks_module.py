import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class MyCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MyCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_rollout_end(self) -> None:
        # Get the mean reward for the last episode
        self.episode_rewards.append(1)
    
    def _on_step(self)-> None:
        print(self.logger)
    
