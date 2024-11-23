# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "gym",
#     "numpy",
# ]
# ///

"""_summary_
"""

import gym
import numpy as np

# Set up the environment (you can replace "FrozenLake-v1" with another environment)
env = gym.make("FrozenLake-v1", is_slippery=True)
n_actions = env.action
