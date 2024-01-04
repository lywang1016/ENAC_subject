import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    T.nn.init.orthogonal_(layer.weight, std)
    T.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorNetwork(nn.Module):
    def __init__(self, actions_dim, input_dims, alpha, agent_name):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(os.getcwd(), 'model', agent_name+'_actor_checkpoint.pth')
        self.best_file = os.path.join(os.getcwd(), 'model', agent_name+'_actor_best.pth')

        self.actor = nn.Sequential(
                layer_init(nn.Linear(input_dims, 1024)),
                # nn.BatchNorm1d(1024),
                # nn.ReLU(),
                nn.Tanh(),
                layer_init(nn.Linear(1024, 256)),
                # nn.BatchNorm1d(256),
                # nn.ReLU(),
                nn.Tanh(),
                layer_init(nn.Linear(256, 32)),
                # nn.BatchNorm1d(32),
                # nn.ReLU(),
                nn.Tanh(),
                layer_init(nn.Linear(32, actions_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(T.zeros(1, actions_dim))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, eps=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        action_mean = self.actor(state)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = T.exp(action_logstd)
        return action_mean, action_std

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        T.save(self.state_dict(), self.best_file)

    def load_best(self):
        self.load_state_dict(T.load(self.best_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, agent_name):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(os.getcwd(), 'model', agent_name+'_critic_checkpoint.pth')
        self.best_file = os.path.join(os.getcwd(), 'model', agent_name+'_critic_best.pth')

        self.critic = nn.Sequential(
                layer_init(nn.Linear(input_dims, 2048)),
                # nn.BatchNorm1d(2048),
                # nn.ReLU(),
                nn.Tanh(),
                layer_init(nn.Linear(2048, 512)),
                # nn.BatchNorm1d(512),
                # nn.ReLU(),
                nn.Tanh(),
                layer_init(nn.Linear(512, 64)),
                # nn.BatchNorm1d(64),
                # nn.ReLU(),
                nn.Tanh(),
                layer_init(nn.Linear(64, 8)),
                # nn.BatchNorm1d(8),
                # nn.ReLU(),
                nn.Tanh(),
                layer_init(nn.Linear(8, 1), std=1.0),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, eps=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        T.save(self.state_dict(), self.best_file)

    def load_best(self):
        self.load_state_dict(T.load(self.best_file))