import os
import torch as T
import torch.nn as nn
import torch.optim as optim

import os
import torch as T
import torch.nn as nn
import torch.optim as optim

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, agent_name):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(os.getcwd(), 'model', agent_name+'_actor.pth')

        self.actor = nn.Sequential(
                nn.Linear(input_dims, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
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

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, agent_name):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(os.getcwd(), 'model', agent_name+'_critic.pth')

        self.critic = nn.Sequential(
                nn.Linear(input_dims, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        temp = T.cat([state, action], dim=1)
        value = self.critic(temp)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))