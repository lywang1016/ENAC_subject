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
                nn.Linear(input_dims, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
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
                nn.Linear(input_dims, 2048),
                nn.ReLU(),
                nn.Linear(2048, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Linear(64, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
        )
        self.output_linear = nn.Linear(32, 1)
        nn.init.ones_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.output_linear(self.critic(state))
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        T.save(self.state_dict(), self.best_file)

    def load_best(self):
        self.load_state_dict(T.load(self.best_file))