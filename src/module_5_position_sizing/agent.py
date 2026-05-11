"""Q-learning agent for position sizing (Module 5).

Maintains a flat Q-table over the discretized MDP state space.
Supports epsilon-greedy exploration during training and greedy
action selection at inference.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .mdp import (
    NUM_ACTIONS,
    NUM_STATES,
    POSITION_PCTS,
    PositionAction,
    State,
    compute_reward,
    state_index,
)


@dataclass
class QAgentConfig:
    """Hyperparameters for the Q-learning agent."""

    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995


class PositionSizingAgent:
    """Tabular Q-learning agent over the position-sizing MDP."""

    def __init__(self, config: QAgentConfig | None = None) -> None:
        self.config = config or QAgentConfig()
        self.q_table = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
        self._epsilon = self.config.epsilon

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def select_action(self, state: State, *, greedy: bool = False) -> PositionAction:
        """Epsilon-greedy (training) or greedy (inference) action selection."""
        si = state_index(state)
        if not greedy and random.random() < self._epsilon:
            return PositionAction(random.randint(0, NUM_ACTIONS - 1))
        return PositionAction(int(np.argmax(self.q_table[si])))

    def update(
        self,
        state: State,
        action: PositionAction,
        reward: float,
        next_state: State,
    ) -> float:
        """One-step Q-learning update. Returns the TD error."""
        si = state_index(state)
        nsi = state_index(next_state)
        best_next = float(np.max(self.q_table[nsi]))
        td_target = reward + self.config.gamma * best_next
        td_error = td_target - self.q_table[si, action]
        self.q_table[si, action] += self.config.alpha * td_error
        return float(td_error)

    def decay_epsilon(self) -> None:
        self._epsilon = max(self.config.epsilon_min,
                            self._epsilon * self.config.epsilon_decay)

    def q_values_for(self, state: State) -> dict[PositionAction, float]:
        """Return Q-values for every action in the given state."""
        si = state_index(state)
        return {PositionAction(a): float(self.q_table[si, a]) for a in range(NUM_ACTIONS)}

    def train_episode(
        self,
        states: list[State],
        strategy_return: float,
        max_drawdown: float,
        volatility: float,
    ) -> float:
        """Run one training episode over a sequence of states.

        Each state is treated as a step where the agent picks a position size,
        receives a reward, and transitions to the next state.  Returns total
        episode reward.
        """
        if not states:
            return 0.0

        total_reward = 0.0
        for i, s in enumerate(states):
            action = self.select_action(s)
            pct = POSITION_PCTS[action]
            reward = compute_reward(pct, strategy_return, max_drawdown, volatility)
            next_s = states[i + 1] if i + 1 < len(states) else s
            self.update(s, action, reward, next_s)
            total_reward += reward

        self.decay_epsilon()
        return total_reward
