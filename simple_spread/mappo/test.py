import os
import time
from os.path import exists
from pettingzoo.mpe import simple_spread_v3
from agent import Agent

N_GAMES = 5

max_cycles = 25
env = simple_spread_v3.parallel_env(N=2, max_cycles=max_cycles, render_mode="human")
observations, infos = env.reset()
n_actions = 5
# all_states_dims = 56 # 3 agents, 18+19+19
# observation_dims = 18
all_states_dims = 25 # 2 agents, 12+13
observation_dims = 12

flag = True
agents = {}
for name in observations:
    agents[name] = Agent(name, n_actions, observation_dims, all_states_dims)
    actor_model_checkpoint_path = os.path.join(os.getcwd(), 'model', name+'_actor_best.pth')
    critic_model_checkpoint_path = os.path.join(os.getcwd(), 'model', name+'_critic_best.pth')
    # actor_model_checkpoint_path = os.path.join(os.getcwd(), 'model', name+'_actor_checkpoint.pth')
    # critic_model_checkpoint_path = os.path.join(os.getcwd(), 'model', name+'_critic_checkpoint.pth')
    if exists(actor_model_checkpoint_path) and exists(critic_model_checkpoint_path):
        print("Agent " + name + " Load Model")
        agents[name].load_best()
        # agents[name].load_checkpoints()
    else:
        print("Agent " + name + " No Model")
        flag = False

if flag:
    for i in range(N_GAMES):
        print("Game " + str(i))
        obs, infos = env.reset()
        flag = False
        while not flag:
            actions = {}
            for agent_name in observations:
                action = agents[agent_name].deterministic_action(observations[agent_name]) 
                # action, probs_old = agents[agent_name].stochastic_action(observations[agent_name]) 
                actions[agent_name] = action
            observations, rewards, terminations, truncations, infos = env.step(actions)
            time.sleep(0.1) # to slow down the action for the video
            for name in obs:
                flag = terminations[name] or truncations[name]
env.close()