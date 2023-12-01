import sys 
sys.path.append("..") 
import yaml
import numpy as np
from env import Environment
from agent import Agent

if __name__ == '__main__':
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    N_GAMES = 2

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
    obs = env.reset()
    n_actions = {}
    actor_dims = {}
    critic_dims = 0
    for name in obs:
        n_actions[name] = 5
        actor_dims[name] = len(obs[name])
        critic_dims += actor_dims[name] + n_actions[name]

    agents = {}
    for name in obs:
        agents[name] = Agent(name, n_actions[name], actor_dims[name], critic_dims)
        agents[name].load_models()

    for i in range(N_GAMES):
        print("Game " + str(i))
        obs = env.reset()
        while not env.finish:
            actions_take = {}
            actions = {}
            for name in obs:
                actions[name] = agents[name].choose_action(obs[name])
                actions_take[name] = np.argmax(actions[name])

            obs, rewards, dones = env.step(actions_take)