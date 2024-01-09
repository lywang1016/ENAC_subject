import sys 
sys.path.append("..") 
import yaml
import os
import copy
import numpy as np
import torch as T
from agent import Agent
from env import Environment
from os.path import exists
from utils import plot_learning_curve

if not os.path.exists('model'): 
    os.mkdir('model')
if not os.path.exists('plots'): 
    os.mkdir('plots')

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

env_size = (config['env_length'], config['env_width'])
n_hider = config['n_hider']
n_searcher = config['n_searcher']
dt = config['dt'] 
max_step = config['max_step']
queue_maxlen = config['queue_maxlen']
queue_takelen = config['queue_takelen']
history_len = config['history_len']
hider_size = config['hider_size']
hider_search_range = config['hider_search_range']
hider_max_vel = config['hider_max_vel']
searcher_size = config['searcher_size']
searcher_search_range = config['searcher_search_range']
searcher_max_vel = config['searcher_max_vel']
render_mode = ''

env = Environment(env_size, dt, render_mode, n_hider, n_searcher, 
                max_step, queue_maxlen, queue_takelen, history_len,
                hider_size, hider_search_range, hider_max_vel, 
                searcher_size, searcher_search_range, searcher_max_vel)
observations = env.reset()
actions_dim = 2
observation_dims = (9*(n_hider+n_searcher) + 0) * history_len
all_states_dims = observation_dims*(n_hider+n_searcher) + actions_dim*(n_hider+n_searcher-1)

env_seed = 0
T.manual_seed(0)
T.cuda.manual_seed(0)
T.backends.cudnn.deterministic = True
T.backends.cudnn.benchmark = False

A_LR = 1e-4
C_LR = 1e-4
GAMMA = 0.99
EPS = 0.2
A_BATCH_SIZE = 128
C_BATCH_SIZE = 128
L2_REG = 1e-3
EPOCH = 10
T_LEN = 4096
GAE = 0.95
ENTROPY_COEF = 1e-3
ENTROPY_COEF_DECAY = 0.99

Max_train_steps = 5e5

agents = {}
score_history = {}
best_average_episode_score = {}
for name in observations:
    score_history[name] = []
    best_average_episode_score[name] = -10000
    agents[name] = Agent(name, actions_dim, observation_dims, all_states_dims,
                         A_LR, C_LR, GAMMA, EPS, A_BATCH_SIZE, C_BATCH_SIZE, 
                         L2_REG, EPOCH, T_LEN, GAE, ENTROPY_COEF, ENTROPY_COEF_DECAY)
    actor_model_checkpoint_path = os.path.join(os.getcwd(), 'model', name+'_actor_checkpoint.pth')
    critic_model_checkpoint_path = os.path.join(os.getcwd(), 'model', name+'_critic_checkpoint.pth')
    if exists(actor_model_checkpoint_path) and exists(critic_model_checkpoint_path):
        print("Agent " + name + " Load Checkpoint")
        agents[name].load_checkpoints()
    else:
        print("Agent " + name + " Train From Scratch")

traj_lenth= 0
total_steps = 0
episode = 0

while total_steps < Max_train_steps:
    observations = env.reset()
    done = False
    score = {}
    all_states = copy.deepcopy(observations)
    for name in observations:
        score[name] = 0
        for temp_name in observations:
            if temp_name != name:
                all_states[name] = np.append(all_states[name], observations[temp_name])
                all_states[name] = np.append(all_states[name], np.array([0, 0]))

    '''Interact & trian'''
    while not done:
        '''Interact with Env'''
        actions = {}
        probs_olds = {}
        for name in observations:
            a, logprob_a = agents[name].stochastic_action(observations[name])
            actions[name] = a 
            probs_olds[name] = logprob_a 
        observations_, r, dw, tr= env.step(actions) # dw: dead&win; tr: truncated
        for name in observations:
            score[name] += r[name]   
            done = (dw[name] or tr[name])

        '''Store the current transition'''
        all_states_ = copy.deepcopy(observations_)
        for name in all_states_:
            for temp_name in observations_:
                if temp_name != name:
                    all_states_[name] = np.append(all_states_[name], observations_[temp_name])
                    all_states_[name] = np.append(all_states_[name], np.array(actions[temp_name]))
        for name in observations_:
            agents[name].put_data(all_states[name], observations[name], actions[name], r[name], all_states_[name], \
                                  probs_olds[name], done, dw[name], idx = traj_lenth)
        observations = observations_
        all_states = all_states_
        traj_lenth += 1
        total_steps += 1

        '''Update if its time'''
        if traj_lenth % T_LEN == 0:
            for name in agents:
                agents[name].train()
                agents[name].save_checkpoints()
            traj_lenth = 0

    episode += 1
    for name in agents:
        score_history[name].append(score[name])
        if episode % 100 == 0:
            temp_score_history = score_history[name][(episode-10) : episode]
            ave_score = sum(temp_score_history) / 10
            print("Episode: " + str(episode) + " Step " + str(total_steps) + '/' + str(Max_train_steps) \
                  + ' ' + name + ' average score: ' + str(ave_score))
            if ave_score > best_average_episode_score[name]:
                best_average_episode_score[name] = ave_score
                agents[name].save_best()
                print("Save Best Model of agent " + name)


for name in score_history:
    x = [i+1 for i in range(len(score_history[name]))]
    plot_learning_curve(x, score_history[name], os.path.join(os.getcwd(), 'plots', name+'_rewards.png'))