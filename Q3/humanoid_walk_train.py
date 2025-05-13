import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dmc import make_dmc_env
from collections import deque
import random

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Network definitions
# ─────────────────────────────────────────────────────────────────────────────
def mlp(in_dim, out_dim, hidden=256, final_tanh=False):
    layers = [
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim)
    ]
    if final_tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_limit):
        super().__init__()
        self.net = mlp(state_dim, action_dim, final_tanh=True)
        self.action_limit = action_limit

    def forward(self, state):
        return self.net(state) * self.action_limit

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = mlp(state_dim + action_dim, 1)
        self.q2 = mlp(state_dim + action_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

# ─────────────────────────────────────────────────────────────────────────────
# TD3 Algorithm
# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, size=1_000_000):
        self.buffer = deque(maxlen=size)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)


class TD3Agent:
    def __init__(self, state_dim, action_dim, action_limit):
        self.device = device

        self.actor = Actor(state_dim, action_dim, action_limit).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, action_limit).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay = ReplayBuffer()
        self.action_limit = action_limit

        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2
        self.total_it = 0

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy()[0]
        if noise:
            action += noise * np.random.randn(len(action))
        return np.clip(action, -self.action_limit, self.action_limit)

    def update(self, batch_size=256):
        if len(self.replay) < batch_size:
            return

        self.total_it += 1
        state, action, reward, next_state, done = self.replay.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.action_limit, self.action_limit)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.gamma * torch.min(target_Q1, target_Q2)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.q1(torch.cat([state, self.actor(state)], dim=1)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train():
    env = make_dmc_env("humanoid-walk", seed=np.random.randint(0, 1000000), flatten=True, use_pixels=False)

    state_dim = env.observation_space.shape[0]  # 67
    action_dim = env.action_space.shape[0]      # 21
    action_limit = float(env.action_space.high[0])

    agent = TD3Agent(state_dim, action_dim, action_limit)

    episodes = 10000
    max_steps = 10000
    warmup_steps = 10_000
    total_steps = 0
    eval_interval = 100
    best_score = -float("inf")

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay.push((state, action, reward, next_state, float(done)))
            state = next_state
            episode_reward += reward
            total_steps += 1

            agent.update()

            if done:
                break

        print(f"Episode {episode} | Reward: {episode_reward:.2f}")

        if (episode + 1) % eval_interval == 0:
            score = evaluate(agent)
            print(f"[eval @ ep {episode+1}] average score: {score:.2f}")
            if score > best_score:
                best_score = score
                torch.save(agent.actor.state_dict(), "best_humanoid_actor.pth")

    env.close()

@torch.no_grad()
def evaluate(actor, n_episodes=20):
    env = make_dmc_env("humanoid-walk", seed=np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    scores = []
    for _ in range(n_episodes):
        s, _ = env.reset(seed=np.random.randint(0, 1_000_000))
        ep_ret, done = 0.0, False
        while not done:
            a = actor.actor(torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0))

            a = a.cpu().numpy()[0]
            s, r, term, trunc, _ = env.step(a)
            ep_ret += r
            done = term or trunc
        scores.append(ep_ret)
    env.close()
    mean, std = np.mean(scores), np.std(scores)
    return float(mean - std) 


if __name__ == "__main__":
    train()
