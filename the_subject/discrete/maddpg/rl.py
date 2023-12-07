import sys 
sys.path.append("..") 
import yaml
import os
import numpy as np
import torch as T
from agent import Agent
from memory import MultiAgentReplayBuffer
from utils import plot_learning_curve
from env import Environment

if __name__ == '__main__':
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    N_GAMES = 20000
    GAMMA = 1.0
    LR_ALPHA = 1e-4
    LR_BETA = 1e-4
    TAU = 0.01
    BATCH_SIZE = 2048
    MAX_BUFFER_SIZE = 10000
    PRINT_INTERVAL = 500

    env_size = (config['env_length'], config['env_width'])
    n_hider = config['n_hider']
    n_searcher = config['n_searcher']
    dt = config['dt'] 
    max_step = config['max_step']
    hider_size = config['hider_size']
    hider_search_range = config['hider_search_range']
    hider_max_vel = config['hider_max_vel']
    searcher_size = config['searcher_size']
    searcher_search_range = config['searcher_search_range']
    searcher_max_vel = config['searcher_max_vel']
    render_mode = ''

    env = Environment(env_size, dt, render_mode, n_hider, n_searcher, max_step, 
                      hider_size, hider_search_range, hider_max_vel, 
                      searcher_size, searcher_search_range, searcher_max_vel)
    obs = env.reset()
    n_actions = {}
    actor_dims = {}
    critic_dims = 0
    n_agents = 0
    names = []
    for name in obs:
        n_agents += 1
        names.append(name)
        n_actions[name] = 5
        actor_dims[name] = len(obs[name])
        critic_dims += actor_dims[name] + n_actions[name]
    
    memory = MultiAgentReplayBuffer(MAX_BUFFER_SIZE, actor_dims, n_actions, batch_size=BATCH_SIZE)

    agents = {}
    for name in obs:
        agents[name] = Agent(name, n_actions[name], actor_dims[name], critic_dims,
                            alpha=LR_ALPHA, beta=LR_BETA, gamma=GAMMA, tau=TAU)
    
    total_steps = 0    
    train_cnt = 0
    score_history = {}
    for name in obs:
        score_history[name] = []

    for i in range(N_GAMES):
        obs = env.reset()
        score = {}
        for name in obs:
            score[name] = 0
            
        while not env.finish:
            actions_take = {}
            actions = {}
            for name in obs:
                actions[name] = agents[name].choose_action_with_noise(obs[name])
                actions_take[name] = np.argmax(actions[name])

            obs_, rewards, dones = env.step(actions_take)

            for name in obs_:
                score[name] += rewards[name]

            memory.store_transition(obs, actions, rewards, obs_, dones)

            if total_steps % 30 == 0 and memory.ready():
                actor_states, actions, rewards, actor_new_states, dones = memory.sample_buffer()

                all_states = []
                all_states_ = []
                all_agents_new_actions = []
                all_agents_mu_actions = []
                all_agents_true_actions = []
                for name in agents:
                    temp = T.tensor(np.array(actor_states[name]), dtype=T.float).to(agents[name].actor.device)
                    all_states.append(temp)

                    temp = T.tensor(np.array(actor_new_states[name]), dtype=T.float).to(agents[name].actor.device)
                    all_states_.append(temp)

                    mu_states = T.tensor(actor_states[name], 
                                 dtype=T.float).to(agents[name].actor.device)
                    pi = agents[name].actor.forward(mu_states)
                    all_agents_mu_actions.append(pi)

                    new_states = T.tensor(actor_new_states[name], 
                                 dtype=T.float).to(agents[name].actor.device)
                    new_pi = agents[name].target_actor.forward(new_states)
                    all_agents_new_actions.append(new_pi)

                    temp = T.tensor(np.array(actions[name]), dtype=T.float).to(agents[name].actor.device)
                    all_agents_true_actions.append(temp)
                states = T.cat([acts for acts in all_states], dim=1)
                states_ = T.cat([acts for acts in all_states_], dim=1)
                mu_actions = T.cat([acts for acts in all_agents_mu_actions], dim=1)
                new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
                true_actions = T.cat([acts for acts in all_agents_true_actions],dim=1)

                # only train 1 agent to avoid Error of PyTorch: "one of the variables needed for gradient computation has been modified by an inplace operation"
                agents[names[train_cnt%n_agents]].learn(states, states_, mu_actions, new_actions, 
                                    true_actions, rewards[name], dones[name])
                agents[names[train_cnt%n_agents]].update_network_parameters()
                train_cnt += 1

            obs = obs_
            total_steps += 1
        
        for name in agents:
            score_history[name].append(score[name])
            avg_score = np.mean(score_history[name][-100:])
            agents[name].save_models()
            if i % PRINT_INTERVAL == 0 and i > 0:
                print('episode', i, ' agent: ' + name + ' average score {:.1f}'.format(avg_score))

    for name in score_history:
        x = [i+1 for i in range(len(score_history[name]))]
        plot_learning_curve(x, score_history[name], os.path.join(os.getcwd(), 'plots', name+'_rewards.png'))
    