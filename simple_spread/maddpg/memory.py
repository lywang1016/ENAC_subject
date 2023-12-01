import numpy as np

class MultiAgentReplayBuffer:
    def __init__(self, max_size, actor_dims, n_actions, batch_size):
        n_agents = len(actor_dims)
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.agent_names = []

        self.actor_state_memory = {}
        self.actor_new_state_memory = {}
        self.actor_action_memory = {}
        self.reward_memory = {}
        self.terminal_memory = {}
        for name in actor_dims:
            self.agent_names.append(name)
            self.actor_state_memory[name] = np.zeros((self.mem_size, actor_dims[name]))
            self.actor_new_state_memory[name] = np.zeros((self.mem_size, actor_dims[name]))
            self.actor_action_memory[name] = np.zeros((self.mem_size, n_actions[name]))
            self.reward_memory[name] = np.zeros((self.mem_size, 1))
            self.terminal_memory[name] = np.zeros((self.mem_size, 1), dtype=bool)

    def store_transition(self, raw_obs, action, reward, raw_obs_, done):
        index = self.mem_cntr % self.mem_size

        for name in self.agent_names:
            self.actor_state_memory[name][index] = raw_obs[name]
            self.actor_new_state_memory[name][index] = raw_obs_[name]
            self.actor_action_memory[name][index] = action[name] 
            self.reward_memory[name][index] = reward[name]  
            self.terminal_memory[name][index] = done[name]

        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        actor_states = {}
        actor_new_states = {}
        actions = {}
        rewards = {}
        terminal = {} 
        for name in self.agent_names:
            actor_states[name] = self.actor_state_memory[name][batch]
            actor_new_states[name] = self.actor_new_state_memory[name][batch]
            actions[name] = self.actor_action_memory[name][batch]
            rewards[name] = self.reward_memory[name][batch]
            terminal[name] = self.terminal_memory[name][batch]

        return actor_states, actions, rewards, actor_new_states, terminal

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True