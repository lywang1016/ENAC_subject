import os
import copy
import numpy as np
from tqdm import tqdm
from memory import Trajectory
from agent import Agent
from pettingzoo.mpe import simple_spread_v3
from os.path import exists
from utils import plot_learning_curve

def obs_dict_to_state_vector(observation):
    state = np.array([])
    for name in observation:
        state = np.concatenate([state, observation[name]])
    return state

GPI_LOOP = 500
GAMMA = 0.5
LR = 1e-4
EPS = 0.2
EVALUATION_EPOCH = 5
IMPROVEMENT_EPOCH = 2
BATCH_SIZE = 32
BOOTSTRAPPING = 16
max_cycles = 25
max_episode_num = 256

env = simple_spread_v3.parallel_env(N=2, max_cycles=max_cycles, render_mode="")
observations, infos = env.reset()
n_actions = 5
# all_states_dims = 56 # 3 agents, 18+19+19
# observation_dims = 18
all_states_dims = 25 # 2 agents, 12+13
observation_dims = 12

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
    total_score = {}
    for name in agents:
        agents[name].trajectories = []
        total_score[name] = 0
    episode_num = 0
    while episode_num < max_episode_num:
        done = False
        score = {}
        observations, infos = env.reset()
        # all_states = obs_dict_to_state_vector(observations)
        all_states = copy.deepcopy(observations)
        trajectory = {}
        for name in observations:
            trajectory[name] = Trajectory()  
            score[name] = 0
            for temp_name in observations:
                if temp_name != name:
                    all_states[name] = np.append(all_states[name], observations[temp_name])
                    all_states[name] = np.append(all_states[name], 0)
        while not done:
            actions = {}
            probs_olds = {}
            for name in observations:
                action, probs_old = agents[name].stochastic_action(observations[name]) 
                actions[name] = action
                probs_olds[name] = probs_old
            observations_, rewards, terminations, truncations, infos = env.step(actions)
            # all_states_ = obs_dict_to_state_vector(observations_)
            all_states_ = copy.deepcopy(observations_)
            for name in all_states_:
                for temp_name in observations_:
                    if temp_name != name:
                        all_states_[name] = np.append(all_states_[name], observations_[temp_name])
                        all_states_[name] = np.append(all_states_[name], actions[temp_name])
            for name in observations_:
                if terminations[name] or truncations[name]:
                    done = True
                score[name] += rewards[name]
                trajectory[name].remember(all_states[name], observations[name], actions[name], rewards[name], done, probs_olds[name])
                # trajectory[name].remember(all_states, observations[name], actions[name], rewards[name], done, probs_olds[name])
            observations = observations_
            all_states = all_states_
        episode_num += 1
        for name in trajectory:
            agents[name].trajectories.append(trajectory[name]) 
            score_history[name].append(score[name])
            total_score[name] += score[name]
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

env.close()
for name in score_history:
    x = [i+1 for i in range(len(score_history[name]))]
    plot_learning_curve(x, score_history[name], os.path.join(os.getcwd(), 'plots', name+'_rewards.png'))