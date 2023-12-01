import numpy as np
import torch as T
import torch.nn.functional as F
from network import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, agent_name, n_actions, observation_dim, all_states_dim,
                    alpha=0.01, beta=0.01, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.name = agent_name
        self.actor = ActorNetwork(n_actions, observation_dim, alpha, agent_name+'_policy')
        self.critic = CriticNetwork(all_states_dim, beta, agent_name+'_policy')
        self.target_actor = ActorNetwork(n_actions, observation_dim, alpha, agent_name+'_target')
        self.target_critic = CriticNetwork(all_states_dim, beta, agent_name+'_target')

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_net_state_dict = self.target_actor.state_dict()
        policy_net_state_dict = self.actor.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
        self.target_actor.load_state_dict(target_net_state_dict)

        target_net_state_dict = self.target_critic.state_dict()
        policy_net_state_dict = self.critic.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
        self.target_critic.load_state_dict(target_net_state_dict)

    def choose_action_with_noise(self, observation):
        state = T.tensor(np.array(observation), dtype=T.float).to(self.actor.device)
        with T.no_grad():
            temp = self.actor(state)
        noise = T.rand(self.n_actions).to(self.actor.device)
        action = temp + noise
        action = action.clip(min=0.0, max=1.0)

        return action.detach().cpu().numpy()
    
    def choose_action(self, observation):
        state = T.tensor(np.array(observation), dtype=T.float).to(self.actor.device)
        with T.no_grad():
            action = self.actor(state)

        return action.detach().cpu().numpy()
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
        # print("\t Model Saved!")

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
        print("\t Model Loaded!")

    def learn(self, states, states_, mu_actions, new_actions, true_actions, reward, done):
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device).flatten()
        done = T.tensor(done).to(self.actor.device).flatten()

        critic_value_ = self.target_critic.forward(states_, new_actions).flatten()
        critic_value_[done] = 0.0
        critic_value = self.critic.forward(states, true_actions).flatten()

        target = reward + self.gamma*critic_value_
        critic_loss = F.mse_loss(target, critic_value)
        self.critic.optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic.optimizer.step()

        actor_loss = self.critic.forward(states, mu_actions).flatten()
        actor_loss = -T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()
