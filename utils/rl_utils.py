"""
Shared utility functions used across chapter notebooks.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Optional


def plot_learning_curve(rewards: List[float], window: int = 50, title: str = "Learning Curve",
                        baseline: Optional[float] = None, figsize=(10, 4)):
    """Plot a smoothed learning curve with optional baseline."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(rewards, alpha=0.2, color='steelblue', linewidth=0.5, label='Episode reward')
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(rewards)), smoothed, color='steelblue',
            linewidth=2, label=f'{window}-episode moving average')
    if baseline is not None:
        ax.axhline(y=baseline, color='red', linestyle='--',
                   label=f'Baseline ({baseline:.1f})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def compute_returns(rewards: List[float], gamma: float = 0.99) -> torch.Tensor:
    """Compute discounted returns G_t for a trajectory."""
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
    return (returns - returns.mean()) / (returns.std() + 1e-8)


def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """Compute GAE advantages for an episode."""
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        next_val = 0.0 if dones[t] else next_values[t]
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * (0.0 if dones[t] else gae)
        advantages[t] = gae
    adv = torch.FloatTensor(advantages)
    return (adv - adv.mean()) / (adv.std() + 1e-8)


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """DPO loss function (Chapter 16)."""
    import torch.nn.functional as F
    chosen_ratio   = policy_chosen_logps   - ref_chosen_logps
    rejected_ratio = policy_rejected_logps - ref_rejected_logps
    logits = beta * (chosen_ratio - rejected_ratio)
    loss = -F.logsigmoid(logits).mean()
    accuracy = (chosen_ratio > rejected_ratio).float().mean()
    margin = (chosen_ratio - rejected_ratio).mean()
    return loss, accuracy, margin


def grpo_advantages(rewards: list, eps: float = 1e-8) -> torch.Tensor:
    """GRPO group-relative advantages (Chapter 17)."""
    r = torch.FloatTensor(rewards)
    return (r - r.mean()) / (r.std() + eps)


class ReplayBuffer:
    """Experience replay buffer for DQN (Chapter 5)."""
    from collections import deque

    def __init__(self, capacity: int):
        from collections import deque
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        import random
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)
