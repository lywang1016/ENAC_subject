import time
import numpy as np
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.mpe import simple_v3
from agent import Agent

if __name__ == '__main__':
    MAX_STEPS = 25
    N_GAMES = 5

    env = simple_spread_v3.parallel_env(N=2, max_cycles=MAX_STEPS, render_mode="human", continuous_actions=False)
    # env = simple_v3.parallel_env(max_cycles=MAX_STEPS, render_mode="human", continuous_actions=False)
    obs, infos = env.reset()
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
        obs, infos = env.reset()
        flag = False
        while not flag:
            actions_take = {}
            actions = {}
            for name in obs:
                actions[name] = agents[name].choose_action(obs[name])
                actions_take[name] = np.argmax(actions[name])

            obs, rewards, terminations, truncations, infos = env.step(actions_take)
            time.sleep(0.1) # to slow down the action for the video

            for name in obs:
                flag = terminations[name] or truncations[name]

    env.close()