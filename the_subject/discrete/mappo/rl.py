import sys 
sys.path.append("..") 
import yaml
import os
import copy
import numpy as np
from tqdm import tqdm
from memory import Trajectory
from agent import Agent
from env import Environment
from os.path import exists
from utils import plot_learning_curve

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

GPI_LOOP = 40
GAMMA = 1.0
LR = 1e-4
EPS = 0.2
EVALUATION_EPOCH = 5
IMPROVEMENT_EPOCH = 2
BATCH_SIZE = 1024
BOOTSTRAPPING = 1
MEMORY_SIZE = 16384

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
observations = env.reset()
n_actions = 5
observation_dims = 9*(n_hider+n_searcher)
all_states_dims = observation_dims*(n_hider+n_searcher) + (n_hider+n_searcher-1)

agents = {}
score_history = {}
best_average_episode_score = {}
for name in observations:
    score_history[name] = []
    best_average_episode_score[name] = -10000
    agents[name] = Agent(name, n_actions, observation_dims, all_states_dims,
                            LR, GAMMA, EPS, BATCH_SIZE, BOOTSTRAPPING,
                            EVALUATION_EPOCH, IMPROVEMENT_EPOCH)
    actor_model_checkpoint_path = os.path.join(os.getcwd(), 'model', name+'_actor_checkpoint.pth')
    critic_model_checkpoint_path = os.path.join(os.getcwd(), 'model', name+'_critic_checkpoint.pth')
    if exists(actor_model_checkpoint_path) and exists(critic_model_checkpoint_path):
        print("Agent " + name + " Load Checkpoint")
        agents[name].load_checkpoints()
    else:
        print("Agent " + name + " Train From Scratch")

for i in range(GPI_LOOP):
    print('---------------------- GPI Loop ' + str(i+1) + ' :' + '----------------------')

    print('Generate trajectories...')
    steps = 0
    total_score = {}
    for name in agents:
        agents[name].trajectories = []
        total_score[name] = 0
    episode_num = 0
    while steps < MEMORY_SIZE:
        score = {}
        trajectory = {}
        observations = env.reset()
        all_states = copy.deepcopy(observations)
        for name in observations:
            trajectory[name] = Trajectory()  
            score[name] = 0
            for temp_name in observations:
                if temp_name != name:
                    all_states[name] = np.append(all_states[name], observations[temp_name])
                    all_states[name] = np.append(all_states[name], 0)
        
        while not env.finish:
            actions = {}
            probs_olds = {}
            for name in observations:
                action, probs_old = agents[name].stochastic_action(observations[name]) 
                actions[name] = action
                probs_olds[name] = probs_old
            observations_, rewards, dones = env.step(actions)
            steps += 1
            all_states_ = copy.deepcopy(observations_)
            for name in all_states_:
                for temp_name in observations_:
                    if temp_name != name:
                        all_states_[name] = np.append(all_states_[name], observations_[temp_name])
                        all_states_[name] = np.append(all_states_[name], actions[temp_name])
            for name in observations_:
                score[name] += rewards[name]
                trajectory[name].remember(all_states[name], observations[name], actions[name], rewards[name], env.finish, probs_olds[name])
            observations = observations_
            all_states = all_states_
        
        for name in trajectory:
            agents[name].trajectories.append(trajectory[name]) 
            score_history[name].append(score[name])
            total_score[name] += score[name]
        episode_num += 1

    for name in total_score:
        average_episode_score = total_score[name] / float(episode_num)
        if average_episode_score >= best_average_episode_score[name]:
            best_average_episode_score[name] = average_episode_score 
            agents[name].save_best()
        print("\tAgent " + name + " Average episode score: %.1f" % average_episode_score)
        print("\tAgent " + name + " Best average episode score: %.1f" % best_average_episode_score[name])

    print('Generate memory...')
    for name in agents:
        agents[name].fill_memory()

    print('Evaluation...')
    for name in tqdm(agents):
        agents[name].evaluation()

    print('Improvement...')
    for name in tqdm(agents):
        agents[name].improvement()
        agents[name].save_checkpoints()

for name in score_history:
    x = [i+1 for i in range(len(score_history[name]))]
    plot_learning_curve(x, score_history[name], os.path.join(os.getcwd(), 'plots', name+'_rewards.png'))