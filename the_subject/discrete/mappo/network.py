import os
import torch as T
import torch.nn as nn
import torch.optim as optim

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, agent_name):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(os.getcwd(), 'model', agent_name+'_actor_checkpoint.pth')
        self.best_file = os.path.join(os.getcwd(), 'model', agent_name+'_actor_best.pth')

        self.actor = nn.Sequential(
                nn.Linear(input_dims, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, n_actions),
                nn.Softmax(dim=-1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        return dist

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
                nn.Linear(input_dims, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
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