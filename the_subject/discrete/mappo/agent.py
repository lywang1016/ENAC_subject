import torch as T
import torch.nn as nn
import numpy as np
from network import ActorNetwork, CriticNetwork
from torch.distributions.categorical import Categorical
from memory import Memory
import pandas as pd
import csv
import os
from os.path import exists

class Agent():
    def __init__(self, agent_name, n_actions, observation_dim, all_states_dim,
                 lr=1e-4, gamma=0.99, eps=0.2, batch_size=32, bootstrapping=16,
                 evaluation_epoch=1, improvement_epoch=1):
        self.name = agent_name
        self.n_actions = n_actions
        self.observation_dim = observation_dim
        self.all_states_dim = all_states_dim
        self.gamma = gamma
        self.eps = eps
        self.bootstrapping = bootstrapping
        self.batch_size = batch_size
        self.evaluation_epoch = evaluation_epoch
        self.improvement_epoch = improvement_epoch

        self.memory = Memory(batch_size)
        self.trajectories = []

        self.actor = ActorNetwork(n_actions, observation_dim, lr, agent_name)
        self.critic = CriticNetwork(all_states_dim, lr, agent_name)
        self.criterion = nn.MSELoss().to(self.critic.device)

        skip_first = False
        self.para_path = os.path.join(os.getcwd(), 'model', self.name+'_popartpara.csv')
        if exists(self.para_path):
            with open(self.para_path) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                for row in readCSV:
                    if not skip_first:
                        skip_first = True
                    else:
                        sigma = float(row[0])
                        mu = float(row[1])
                        nu = float(row[2])
            self.sigma = T.tensor(sigma, dtype=T.float).to(self.critic.device)  # consider scalar first
            self.mu = T.tensor(mu, dtype=T.float).to(self.critic.device)
            self.nu = T.tensor(nu, dtype=T.float).to(self.critic.device)
        else:
            self.sigma = T.tensor(1., dtype=T.float).to(self.critic.device)  # consider scalar first
            self.mu = T.tensor(0., dtype=T.float).to(self.critic.device)
            self.nu = self.sigma**2 + self.mu**2 # second-order moment
        self.sigma_new = None
        self.mu_new = None
        self.beta = 10.**(-0.5)
    
    def stochastic_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.actor.device).unsqueeze(0)
        dist = self.actor(state)
        dist = Categorical(dist)
        action = dist.sample()
        probs_old = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        return action, probs_old
    
    def deterministic_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.actor.device).unsqueeze(0)
        probs = self.actor(state)
        action = T.argmax(probs).item()
        return action
    
    def update_popart_para(self, y):
        b = y.shape[0]
        self.mu_new = (1. - self.beta) * self.mu + self.beta * sum(y) / b
        self.nu = (1. - self.beta) * self.nu + self.beta * sum(y**2) / b
        self.sigma_new = T.sqrt(self.nu - self.mu_new**2)

    def update_popart_linear_layer(self):
        relative_sigma = (self.sigma / self.sigma_new)
        self.critic.output_linear.weight.data *= relative_sigma
        self.critic.output_linear.bias.data *= relative_sigma
        self.critic.output_linear.bias.data += (self.mu-self.mu_new)/self.sigma_new
        self.sigma = self.sigma_new
        self.mu = self.mu_new

    def popart_normalize(self, y):
        return (y - self.mu) / self.sigma
    
    def set_train(self):
        self.actor.train()
        self.critic.train()

    def set_eval(self):
        self.actor.eval()
        self.critic.eval() 
    
    def load_checkpoints(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def save_checkpoints(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
    
    def load_best(self):
        self.actor.load_best()
        self.critic.load_best()

    def save_best(self):
        self.actor.save_best()
        self.critic.save_best()

    def fill_memory(self):
        self.memory.clear_memory()
        for trajectory in self.trajectories:
            for i in range(trajectory.length):
                reward_sum = 0
                value_bootstrapping = 0
                discount = 1.0 / self.gamma
                for j in range(self.bootstrapping):
                    discount *= self.gamma
                    if i+j < trajectory.length:
                        reward_sum += discount * trajectory.reward[i+j]
                if i+self.bootstrapping < trajectory.length:
                    if not trajectory.done[i+self.bootstrapping]:
                        state_bootstrapping = trajectory.all_states[i+self.bootstrapping]
                        state = T.tensor(state_bootstrapping, dtype=T.float).to(self.critic.device).unsqueeze(0)
                        with T.no_grad():
                            value_bootstrapping = self.critic(state)
                        value_bootstrapping = T.squeeze(value_bootstrapping).item()
                returns = reward_sum + discount * self.gamma * value_bootstrapping
                self.memory.store_memory(trajectory.all_states[i], trajectory.observation[i], trajectory.action[i], returns, trajectory.probs_old[i])

    def evaluation(self):
        for epoch in range(self.evaluation_epoch):
            all_state_arr, observation_arr, action_arr, returns_arr, probs_old_arr, batches = self.memory.generate_batches()
            for batch in batches:
                if len(batch) == self.batch_size:
                    states = T.tensor(all_state_arr[batch], dtype=T.float).to(self.critic.device)
                    returns = T.tensor(returns_arr[batch], dtype=T.float).to(self.critic.device)
                    self.update_popart_para(returns)
                    self.update_popart_linear_layer()
                    critic_value = self.critic(states)
                    critic_value = T.squeeze(critic_value) 
                    normalize_returns = self.popart_normalize(returns)
                    # critic_loss = self.criterion(returns.view(self.batch_size, 1), critic_value.view(self.batch_size, 1))  
                    critic_loss = self.criterion(normalize_returns.view(self.batch_size, 1), critic_value.view(self.batch_size, 1))   
                    self.critic.optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic.optimizer.step()
        dataframe = pd.DataFrame({'sigma': [self.sigma.item()],
                                  'mu': [self.mu.item()],
                                  'nu': [self.nu.item()]})
        dataframe.to_csv(self.para_path, index=False, sep=',')

    def improvement(self):
        for epoch in range(self.improvement_epoch):
            all_state_arr, observation_arr, action_arr, returns_arr, probs_old_arr, batches = self.memory.generate_batches()
            for batch in batches:
                if len(batch) == self.batch_size:
                    states = T.tensor(all_state_arr[batch], dtype=T.float).to(self.actor.device)
                    observations = T.tensor(observation_arr[batch], dtype=T.float).to(self.actor.device)
                    actions = T.tensor(action_arr[batch], dtype=T.float).to(self.actor.device)
                    returns = T.tensor(returns_arr[batch], dtype=T.float).to(self.actor.device)
                    probs_old = T.tensor(probs_old_arr[batch], dtype=T.float).to(self.actor.device)
                    dist = self.actor(observations)
                    dist = Categorical(dist)
                    probs = dist.log_prob(actions)
                    critic_value = self.critic(states)
                    critic_value = T.squeeze(critic_value)
                    advantage = returns - critic_value
                    prob_ratio = (probs - probs_old).exp()
                    weighted_probs = prob_ratio * advantage
                    weighted_clipped_probs = T.clamp(prob_ratio, 1-self.eps, 1+self.eps) * advantage
                    actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                    self.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor.optimizer.step()