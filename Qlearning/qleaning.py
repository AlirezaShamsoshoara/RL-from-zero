# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "gym",
#     "numpy",
#     "tqdm",
# ]
# ///

"""_summary_"""

import gym
import numpy as np
from tqdm import tqdm

# Set up the environment (you can replace "FrozenLake-v1" with another environment)
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
n_actions = env.action_space.n
n_states = env.observation_space.n


""" Q-learning parameteres """
alpha = 0.1
gamma = 0.99
epsilon = 0.99
episodes = 100000
epsilon_min = 0.01
epsilon_decay = 0.995


q = np.zeros((n_states, n_actions))


def epsilon_greedy(state, epsilon):
    """_summary_

    Args:
        state (_type_): _description_
        epsilon (_type_): _description_

    Returns:
        _type_: _description_
    """
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q[state, :])


# Q-Learning algorithm

for eps in tqdm(range(episodes)):
    state = env.reset()
    if isinstance(state, tuple) and len(state) > 1:
        state = state[0]
    done = False
    eps_reward = 0

    while not done:
        action = epsilon_greedy(state, epsilon=epsilon)
        next_state, reward, done, truncated, _ = env.step(action)
        # print(f"state:{state}, action:{action}, next_state:{next_state}, reward:{reward}, done:{done}")

        best_next_action = np.argmax(q[next_state])
        td_target = reward + gamma * max(q[next_state, :])
        q[state, action] = q[state, action] + alpha * (td_target - q[state, action])

        state = next_state
        eps_reward += reward

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    # print(f"Q-Table after training:\n{q}\n")
    # print(f"EPS reward = {reward}")

test_eps = 5
for eps in range(test_eps):
    state = env.reset()
    if isinstance(state, tuple) and len(state) > 1:
        state = state[0]
    done = False
    print(f"Episode {eps + 1}: ")
    while not done:
        action = np.argmax(q[state])
        state, reward, done, trunc, _ = env.step(action)
        env.render()
    print(f"Final reward = {reward}")
env.close()
