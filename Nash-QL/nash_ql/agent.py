from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from .nash_solver import compute_nash_value, solve_nash_equilibrium


@dataclass
class Transition:
    """Transition for Nash Q-learning (joint actions)."""

    agent: int
    state: int
    joint_action: tuple[int, ...]  # Joint action of all agents
    reward: float
    next_state: int
    done: bool


class NashQLearningAgent:
    """
    Nash Q-Learning agent for 2-player cooperative games.

    Each agent maintains Q-values over joint action space. At each state, agents
    compute a Nash equilibrium of the stage game and act according to the
    equilibrium strategy.

    Reference:
        Hu, J., & Wellman, M. P. (2003). Nash Q-learning for general-sum stochastic
        games. Journal of machine learning research, 4(Nov), 1039-1069.
    """

    def __init__(
        self,
        n_agents: int,
        n_states: int,
        n_actions: int,
        alpha: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
    ):
        """
        Initialize Nash Q-Learning agent.

        Args:
            n_agents: Number of agents (currently only supports 2)
            n_states: Number of states
            n_actions: Number of actions per agent
            alpha: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exponential decay rate for epsilon
        """
        if n_agents != 2:
            raise ValueError("Nash Q-learning currently only supports 2 agents")

        self.n_agents = n_agents
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay
        self.global_step = 0

        # Q[agent][state][action_1][action_2] = Q-value for agent at state with joint action
        self.Q = np.zeros(
            (n_agents, n_states, n_actions, n_actions), dtype=np.float32
        )

    def epsilon(self) -> float:
        """Compute current epsilon value using exponential decay."""
        eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -self.eps_decay * max(self.global_step - 1, 0)
        )
        return float(np.clip(eps, self.eps_end, self.eps_start))

    def act(self, states: Sequence[int]) -> List[int]:
        """
        Select actions for all agents using epsilon-Nash strategy.

        With probability epsilon, explores uniformly. Otherwise, computes Nash
        equilibrium of the stage game at the current state and samples actions
        according to the equilibrium mixed strategies.

        Args:
            states: List of state indices for each agent

        Returns:
            List of action indices for each agent
        """
        if len(states) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} states, got {len(states)}")

        self.global_step += 1
        eps = self.epsilon()

        # Validate states
        for agent_idx, state in enumerate(states):
            if state < 0 or state >= self.n_states:
                raise ValueError(
                    f"State index {state} out of bounds for agent {agent_idx}"
                )

        # Epsilon-greedy exploration
        if np.random.rand() < eps:
            # Uniform random exploration
            return [int(np.random.randint(self.n_actions)) for _ in range(self.n_agents)]

        # Exploit: use Nash equilibrium strategy
        # For multi-agent with joint states, we need to check if all agents
        # observe the same state (fully observable)
        state = states[0]  # Assume agents observe the same joint state

        # Extract payoff matrices for the stage game at this state
        payoff_1 = self.Q[0, state, :, :]  # Agent 0's Q-values
        payoff_2 = self.Q[1, state, :, :]  # Agent 1's Q-values

        # Solve for Nash equilibrium
        pi_1, pi_2 = solve_nash_equilibrium(payoff_1, payoff_2)

        # Sample actions according to Nash equilibrium probabilities
        action_1 = int(np.random.choice(self.n_actions, p=pi_1))
        action_2 = int(np.random.choice(self.n_actions, p=pi_2))

        return [action_1, action_2]

    def greedy_actions(self, states: Sequence[int]) -> List[int]:
        """
        Select deterministic actions using Nash equilibrium (no exploration).

        Args:
            states: List of state indices for each agent

        Returns:
            List of action indices for each agent (sampled from Nash equilibrium)
        """
        if len(states) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} states, got {len(states)}")

        state = states[0]  # Assume fully observable

        # Extract payoff matrices
        payoff_1 = self.Q[0, state, :, :]
        payoff_2 = self.Q[1, state, :, :]

        # Solve for Nash equilibrium
        pi_1, pi_2 = solve_nash_equilibrium(payoff_1, payoff_2)

        # For deterministic policy, choose actions with highest probability
        # (or sample if multiple best actions)
        action_1 = int(np.random.choice(self.n_actions, p=pi_1))
        action_2 = int(np.random.choice(self.n_actions, p=pi_2))

        return [action_1, action_2]

    def update(self, transitions: Iterable[Transition]) -> None:
        """
        Update Q-values using Nash Q-learning rule.

        For each agent i:
            Q_i(s, a) += alpha * (r_i + gamma * V_i^Nash(s') - Q_i(s, a))

        where V_i^Nash(s') is agent i's expected value at s' under Nash equilibrium.

        Args:
            transitions: List of transitions (one per agent)
        """
        transitions_list = list(transitions)

        for tr in transitions_list:
            agent = tr.agent
            state = tr.state
            joint_action = tr.joint_action
            reward = tr.reward
            next_state = tr.next_state
            done = tr.done

            if len(joint_action) != self.n_agents:
                raise ValueError(
                    f"Expected joint action of size {self.n_agents}, got {len(joint_action)}"
                )

            # Compute Nash equilibrium value at next state
            if done:
                nash_value = 0.0
            else:
                # Extract payoff matrices at next state
                payoff_1_next = self.Q[0, next_state, :, :]
                payoff_2_next = self.Q[1, next_state, :, :]

                # Solve Nash equilibrium
                pi_1_next, pi_2_next = solve_nash_equilibrium(
                    payoff_1_next, payoff_2_next
                )

                # Compute expected value for this agent under Nash equilibrium
                v1, v2 = compute_nash_value(
                    payoff_1_next, payoff_2_next, pi_1_next, pi_2_next
                )
                nash_value = v1 if agent == 0 else v2

            # TD update
            a1, a2 = joint_action[0], joint_action[1]
            current_q = self.Q[agent, state, a1, a2]
            target = reward + self.gamma * nash_value
            td_error = target - current_q
            self.Q[agent, state, a1, a2] += self.alpha * td_error
