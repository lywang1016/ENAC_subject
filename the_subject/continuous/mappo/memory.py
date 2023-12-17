import numpy as np

class Trajectory():
    def __init__(self):
        self.all_states = []
        self.observation = []
        self.action = []
        self.reward = []
        self.done = []
        self.probs_old = []
        self.length = 0

    def remember(self, all_states, observation, action, reward, done, probs_old):
        self.length += 1
        self.all_states.append(all_states)
        self.observation.append(observation)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
        self.probs_old.append(probs_old)

class Memory:
    def __init__(self, batch_size):
        self.all_states = []
        self.observation = []
        self.action = []
        self.returns = []
        self.probs_old = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.all_states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.all_states), \
                np.array(self.observation), \
                np.array(self.action), \
                np.array(self.returns), \
                np.array(self.probs_old), \
                batches

    def store_memory(self, all_state, observation, action, returns, probs_old):
        self.all_states.append(all_state)
        self.observation.append(observation)
        self.action.append(action)
        self.returns.append(returns)
        self.probs_old.append(probs_old)

    def clear_memory(self):
        self.all_states = []
        self.observation = []
        self.action = []
        self.returns = []
        self.probs_old = []