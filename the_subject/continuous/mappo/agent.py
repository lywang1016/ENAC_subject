import torch as T
import torch.nn as nn
import numpy as np
from network import ActorNetwork, CriticNetwork
from memory import Memory

class Agent():
    def __init__(self, agent_name, actions_dim, observation_dim, all_states_dim,
                 lr=1e-4, gamma=0.99, eps=0.2, batch_size=32, bootstrapping=16,
                 evaluation_epoch=1, improvement_epoch=1):
        self.name = agent_name
        self.actions_dim = actions_dim
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

        self.actor = ActorNetwork(actions_dim, observation_dim, lr, agent_name)
        self.critic = CriticNetwork(all_states_dim, lr, agent_name)
        self.criterion = nn.MSELoss().to(self.critic.device)

    def stochastic_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu, sigma = self.actor(state)
        distribution = T.distributions.normal.Normal(mu, sigma) 
        a = distribution.sample()
        probs_old = T.squeeze(distribution.log_prob(a)).detach().cpu().numpy()
        action = T.squeeze(a).detach().cpu().numpy()
        return np.clip(action, -1, 1), probs_old
    
    def deterministic_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu, sigma = self.actor(state)
        action = T.squeeze(mu).detach().cpu().numpy()
        return np.clip(action, -1, 1)
    
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
                        state = T.tensor(state_bootstrapping, dtype=T.float).to(self.critic.device)
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
                    critic_value = self.critic(states)
                    critic_value = T.squeeze(critic_value) 
                    critic_loss = self.criterion(returns.view(self.batch_size, 1), critic_value.view(self.batch_size, 1))    
                    self.critic.optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic.optimizer.step()

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
                    mu, sigma = self.actor(observations)
                    distribution = T.distributions.normal.Normal(mu, sigma) 
                    probs = distribution.log_prob(actions)
                    critic_value = self.critic(states)
                    critic_value = T.squeeze(critic_value)
                    advantage = returns - critic_value
                    prob_ratio = (probs - probs_old).exp()
                    temp = []
                    for i in range(prob_ratio.shape[1]):
                        temp.append(prob_ratio[:,i] * advantage)
                    weighted_probs = T.cat(temp, dim=0)
                    clipped_probs_ratio = T.clamp(prob_ratio, 1-self.eps, 1+self.eps)
                    temp = []
                    for i in range(clipped_probs_ratio.shape[1]):
                        temp.append(clipped_probs_ratio[:,i] * advantage)
                    weighted_clipped_probs = T.cat(temp, dim=0)
                    actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                    self.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor.optimizer.step()