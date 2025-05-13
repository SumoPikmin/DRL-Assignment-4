import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


device = "cpu" # cpu because of leaderboard
# Define the MLP network
def mlp(in_dim, out_dim, hidden=256, final_tanh=False):
    layers = [
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim)
    ]
    if final_tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)

# Define the Actor (Policy) Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_limit):
        super().__init__()
        self.net = mlp(state_dim, action_dim, final_tanh=True)
        self.action_limit = action_limit

    def forward(self, state):
        return self.net(state) * self.action_limit

# Define the Agent that uses the actor network
class Agent(object):
    def __init__(self, actor_model_path):
        # Define the action space
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)

        # Load the actor model weights
        self.actor = Actor(state_dim=67, action_dim=21, action_limit=1.0).to(device)
        self.actor.load_state_dict(torch.load(actor_model_path))  # Load weights
        self.actor.eval()  # Set the actor to evaluation mode

    def act(self, observation):
        # Convert the observation to a tensor and pass it through the actor
        state = torch.FloatTensor(observation).unsqueeze(0).to(device)
        action = self.actor(state).cpu().data.numpy()[0]
        
        # Clip action to ensure it's within the valid range
        return np.clip(action, -1.0, 1.0)
