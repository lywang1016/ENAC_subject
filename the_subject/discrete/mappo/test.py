import sys 
sys.path.append("..") 
import yaml
import os
from os.path import exists
from env import Environment
from agent import Agent

N_GAMES = 5
USE_BEST = False

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

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
render_mode = 'human'

env = Environment(env_size, dt, render_mode, n_hider, n_searcher, max_step, 
                  hider_size, hider_search_range, hider_max_vel, 
                  searcher_size, searcher_search_range, searcher_max_vel)
observations = env.reset(show=False)
n_actions = 5
observation_dims = 9*(n_hider+n_searcher)
all_states_dims = observation_dims*(n_hider+n_searcher) + (n_hider+n_searcher-1)

flag = True
agents = {}
for name in observations:
    agents[name] = Agent(name, n_actions, observation_dims, all_states_dims)
    if USE_BEST:
        actor_model_path = os.path.join(os.getcwd(), 'model', name+'_actor_best.pth')
        critic_model_path = os.path.join(os.getcwd(), 'model', name+'_critic_best.pth')
    else:
        actor_model_path = os.path.join(os.getcwd(), 'model', name+'_actor_checkpoint.pth')
        critic_model_path = os.path.join(os.getcwd(), 'model', name+'_critic_checkpoint.pth')
    if exists(actor_model_path) and exists(critic_model_path):
        print("Agent " + name + " Load Model")
        if USE_BEST:
            agents[name].load_best()
        else:
            agents[name].load_checkpoints()
    else:
        print("Agent " + name + " No Model")
        flag = False

if flag:
    for i in range(N_GAMES):
        print('---------------------- Game ' + str(i+1) + ' ----------------------')
        observations = env.reset()
        while not env.finish:
            actions = {}
            for name in observations:
                action = agents[name].deterministic_action(observations[name]) 
                actions[name] = action
            observations, rewards, dones = env.step(actions)