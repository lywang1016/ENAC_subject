import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class Robot:
    def __init__(self, size, search_range, max_vel, 
                 position, env_size):
        self.size = size
        self.search_range = search_range
        self.max_vel = max_vel
        self.pos = position
        self.vel = [0, 0]
        self.env_size = env_size

    def set_pos(self, position):
        self.pos = position
        self.vel = [0, 0]

    def move(self, action, dt): 
        desired_angle = 2*(action[0]-0.5)*np.pi           # action range from 0 to 1         
        vx = np.cos(desired_angle) * self.max_vel 
        vy = np.sin(desired_angle) * self.max_vel 
        self.vel = [vx, vy]
        self.pos[0] += vx * dt
        self.pos[1] += vy * dt
        self.pos[0] = np.clip(self.pos[0], 0, self.env_size[0])
        self.pos[1] = np.clip(self.pos[1], 0, self.env_size[1])

class Environment:
    def __init__(self, env_size, dt, render_mode, n_hider, n_searcher, 
                 max_step, queue_maxlen, queue_takelen, history_len,
                 hider_size, hider_search_range, hider_max_vel, 
                 searcher_size, searcher_search_range, searcher_max_vel):
        self.n_hider = n_hider
        self.n_searcher = n_searcher
        self.env_size = env_size
        self.dt = dt
        self.max_step = max_step 
        self.done = False
        self.truncated = False
        self.life_time = 0
        self.render_mode = render_mode
        self.queue_maxlen = queue_maxlen
        self.queue_takelen = queue_takelen
        self.history_len = history_len
        self.chase_dis = dt * (hider_max_vel - searcher_max_vel)

        self.robots = {}
        # self.pos_history = {}
        self.hider_names = []
        self.searcher_names = []
        self.single_step_obs = deque(maxlen=self.history_len)

        hider_x = np.random.rand(n_hider) * env_size[0]
        searcher_x = np.random.rand(n_searcher) * env_size[0]
        for i in range(n_hider):
            name = 'hider_' + str(i)
            self.hider_names.append(name)
            self.robots[name] = Robot(hider_size, hider_search_range, hider_max_vel, 
                                      [hider_x[i], env_size[1]], env_size)
            # self.pos_history[name] = deque(maxlen=self.queue_maxlen)
            # self.pos_history[name].append((self.robots[name].pos[0], self.robots[name].pos[1]))
        for i in range(n_searcher):
            name = 'searcher_' + str(i)
            self.searcher_names.append(name)
            self.robots[name] = Robot(searcher_size, searcher_search_range, searcher_max_vel, 
                                      [searcher_x[i], 0], env_size)
            # self.pos_history[name] = deque(maxlen=self.queue_maxlen)
            # self.pos_history[name].append((self.robots[name].pos[0], self.robots[name].pos[1]))
        
        self.distances_hider_searcher = self.hider_searcher_distance()
        self.distances_searcher_hider = self.searcher_hider_distance()

        # history_boundary = self.get_history_boundary()
        # self.single_step_obs.append(self.generate_single_step_obs(history_boundary))

        if render_mode == 'human':
            plt.figure(figsize=(7, 7))

    def reset(self, show=True):
        self.done = False
        self.truncated = False
        self.life_time = 0

        hider_x = np.random.rand(self.n_hider) * self.env_size[0]
        searcher_x = np.random.rand(self.n_searcher) * self.env_size[0]
        for i in range(self.n_hider):
            name = 'hider_' + str(i)
            self.robots[name].set_pos([hider_x[i], self.env_size[1]])
            # self.pos_history[name].clear()
            # self.pos_history[name].append((self.robots[name].pos[0], self.robots[name].pos[1]))
        for i in range(self.n_searcher):
            name = 'searcher_' + str(i)
            self.robots[name].set_pos([searcher_x[i], 0])
            # self.pos_history[name].clear()
            # self.pos_history[name].append((self.robots[name].pos[0], self.robots[name].pos[1]))

        self.distances_hider_searcher = self.hider_searcher_distance()
        self.distances_searcher_hider = self.searcher_hider_distance()
        
        # history_boundary = self.get_history_boundary()
        # self.single_step_obs.clear()
        # self.single_step_obs.append(self.generate_single_step_obs(history_boundary))

        if self.render_mode == 'human' and show:
            self.show_env()

        return self.generate_obs()
    
    def step(self, actions):
        # Generate new states for all agents
        for name in self.robots:
            self.robots[name].move(actions[name], self.dt)
            # self.pos_history[name].append((self.robots[name].pos[0], self.robots[name].pos[1]))

        # Check episode end
        self.life_time += 1
        if self.life_time > self.max_step: # check if get max_step
            self.truncated = True
        for hider in self.hider_names: # check if a hider been capture
            for searcher in self.searcher_names:
                dis, visible = self.distance_from_a_look_b(searcher, hider)
                if dis < self.robots[hider].size + self.robots[searcher].size:
                    self.done = True
                    break
            if self.done:
                break

        # Asign rewards
        rewards = {}
        dones = {}
        truncated = {}
        
        for name in self.hider_names:
            dones[name] = self.done
            truncated[name] = self.truncated
            rewards[name] = 1   # reward for life 1 step more
            dis_to_bound = min([self.robots[name].pos[0], self.robots[name].pos[1], \
                                self.env_size[0] - self.robots[name].pos[0], \
                                self.env_size[1] - self.robots[name].pos[1]])
            if dis_to_bound < self.robots[name].size:   # punish for close to the boundary
                rewards[name] -= (self.robots[name].size - dis_to_bound) / self.robots[name].size
        for name in self.searcher_names:
            dones[name] = self.done
            truncated[name] = self.truncated
            rewards[name] = -1  # punish for give hider life 1 step more
            dis_to_bound = min([self.robots[name].pos[0], self.robots[name].pos[1], \
                                self.env_size[0] - self.robots[name].pos[0], \
                                self.env_size[1] - self.robots[name].pos[1]])
            if dis_to_bound < self.robots[name].size:   # punish for close to the boundary
                rewards[name] -= (self.robots[name].size - dis_to_bound) / self.robots[name].size

        # if self.n_hider > 1:
        #     for i in range(self.n_hider-1):
        #         for j in range(i+1, self.n_hider):
        #             dis, visible = self.distance_from_a_look_b(self.hider_names[i], self.hider_names[j])
        #             if dis < self.robots[self.hider_names[i]].size + self.robots[self.hider_names[j]].size:
        #                 rewards[self.hider_names[i]] -= 10      # punish for colision
        #                 rewards[self.hider_names[j]] -= 10
        # if self.n_searcher > 1:
        #     for i in range(self.n_searcher-1):
        #         for j in range(i+1, self.n_searcher):
        #             dis, visible = self.distance_from_a_look_b(self.searcher_names[i], self.searcher_names[j])
        #             if dis < self.robots[self.searcher_names[i]].size + self.robots[self.searcher_names[j]].size:
        #                 rewards[self.searcher_names[i]] -= 10   # punish for colision
        #                 rewards[self.searcher_names[j]] -= 10

        now_distances_hider_searcher = self.hider_searcher_distance()
        for name in self.hider_names:
            total_dis_before = sum(self.distances_hider_searcher[name])
            total_dis_now = sum(now_distances_hider_searcher[name])
            total_dis_change = total_dis_now - total_dis_before
            min_dis_before = min(self.distances_hider_searcher[name])
            min_dis_now = min(now_distances_hider_searcher[name])
            min_dis_change = min_dis_now - min_dis_before
            dis_metric = 3*min_dis_change + total_dis_change
            # rewards[name] += 1*(min_dis_change - self.chase_dis)
            if dis_metric > 0:
                rewards[name] += dis_metric     # reward for get far from searcher
        self.distances_hider_searcher = copy.deepcopy(now_distances_hider_searcher)

        now_distances_searcher_hider = self.searcher_hider_distance()
        for name in self.searcher_names:
            total_dis_before = sum(self.distances_searcher_hider[name])
            total_dis_now = sum(now_distances_searcher_hider[name])
            total_dis_change = total_dis_now - total_dis_before
            min_dis_before = min(self.distances_searcher_hider[name])
            min_dis_now = min(now_distances_searcher_hider[name])
            min_dis_change = min_dis_now - min_dis_before
            dis_metric = 3*min_dis_change + total_dis_change
            # rewards[name] -= 1*(min_dis_change - self.chase_dis)
            # if dis_metric < 0:
            #     rewards[name] -= dis_metric     # reward for get close from hider
            if min_dis_change > self.chase_dis: # punish for get far from hider
                rewards[name] -= 4*(min_dis_change - self.chase_dis)
        self.distances_searcher_hider = copy.deepcopy(now_distances_searcher_hider)

        # history_boundary = self.get_history_boundary()
        # self.single_step_obs.append(self.generate_single_step_obs(history_boundary))
        # for searcher in self.searcher_names:
        #     find_hider = False
        #     for hider in self.hider_names:
        #         dis, visible = self.distance_from_a_look_b(searcher, hider)
        #         if visible:
        #             find_hider = True
        #             break
        #     if not find_hider:
        #         x = self.robots[searcher].pos[0]
        #         y = self.robots[searcher].pos[1]
        #         if x <= history_boundary[searcher][1] and x >= history_boundary[searcher][0] \
        #         and y <= history_boundary[searcher][3] and y >= history_boundary[searcher][2]:
        #             rewards[searcher] -= 5  # punish for stay in recent history area if don't have a hider in search range
        
        if self.render_mode == 'human':
            self.show_env()

        return self.generate_obs(), rewards, dones, truncated
    
    # def generate_obs(self):
    #     length = len(self.single_step_obs)
    #     obs = {}
    #     for name in self.robots:
    #         obs[name] = []
    #         for i in range(self.history_len - length + 1):
    #             obs[name].append(self.single_step_obs[0][name])
    #         for i in range(1, length):
    #             obs[name].append(self.single_step_obs[i][name])
    #         obs[name] = np.array(obs[name]).flatten()
    #     return obs

    def generate_obs(self):
        obs = {}
        for name in self.robots:
            obs[name] = []
            for i in range(self.n_hider):
                temp = self.generate_obs_from_a_look_b(name, self.hider_names[i])
                obs[name].append(temp)
            for i in range(self.n_searcher):
                temp = self.generate_obs_from_a_look_b(name, self.searcher_names[i])
                obs[name].append(temp)
            obs[name] = np.array(obs[name]).flatten()
        return obs
    
    # def generate_single_step_obs(self, history_boundary):
    #     obs = {}
    #     for name in self.robots:
    #         obs[name] = []
    #         # # history min max
    #         # for item in history_boundary[name]:
    #         #     obs[name].append(item)
    #         # observation to each agent
    #         for i in range(self.n_hider):
    #             temp = self.generate_obs_from_a_look_b(name, self.hider_names[i])
    #             for val in temp:
    #                 obs[name].append(val)
    #         for i in range(self.n_searcher):
    #             temp = self.generate_obs_from_a_look_b(name, self.searcher_names[i])
    #             for val in temp:
    #                 obs[name].append(val)
    #         obs[name] = np.array(obs[name]).flatten()
    #     return obs
    
    # def get_history_boundary(self):
    #     res = {}
    #     for name in self.robots:
    #         length = min([len(self.pos_history[name]), self.queue_takelen])
    #         x = []
    #         y = []
    #         for i in range(length):
    #             x.append(self.pos_history[name][i][0])
    #             y.append(self.pos_history[name][i][1])
    #         res[name] = [min(x), max(x), min(y), max(y)]
    #     return res
    
    def hider_searcher_distance(self):
        res = {}
        for hider in self.hider_names:
            res[hider] = []
            for searcher in self.searcher_names:
                temp, visible = self.distance_from_a_look_b(hider, searcher)
                res[hider].append(temp)
        return res
    
    def searcher_hider_distance(self):
        res = {}
        for searcher in self.searcher_names:    
            res[searcher] = []  
            for hider in self.hider_names:
                temp, visible = self.distance_from_a_look_b(searcher, hider)
                res[searcher].append(temp)
        return res
    
    def distance_from_a_look_b(self, a, b):
        dis = np.linalg.norm(x = [self.robots[a].pos[0] - self.robots[b].pos[0], \
                                self.robots[a].pos[1] - self.robots[b].pos[1]], ord = 2)
        if dis < self.robots[a].search_range + self.robots[b].size:
            return dis, True
        else:
            return self.robots[a].search_range + self.robots[b].size + 1e-3, False

    def generate_obs_from_a_look_b(self, a, b):
        dis, visible = self.distance_from_a_look_b(a, b)
        if visible:
            res = [self.robots[b].pos[0], self.robots[b].pos[1], \
                    self.env_size[0] - self.robots[b].pos[0], \
                    self.env_size[1] - self.robots[b].pos[1], \
                    self.robots[b].vel[0], self.robots[b].vel[1], \
                    (self.robots[b].pos[0] - self.robots[a].pos[0]), \
                    (self.robots[b].pos[1] - self.robots[a].pos[1])]
            res.append(dis)
            return res
        else:
            return [-1] * 9
    
    def show_env(self):
        rectangle = plt.Rectangle((0, 0), self.env_size[0], self.env_size[1], fc = 'w', ec = "black")
        plt.gca().add_patch(rectangle)

        for i in range(self.n_hider):
            name = 'hider_' + str(i)
            circle = plt.Circle((self.robots[name].pos[0], self.robots[name].pos[1]), 
                                self.robots[name].size, color='green', fill=True)
            plt.gca().add_patch(circle)
            circle = plt.Circle((self.robots[name].pos[0], self.robots[name].pos[1]), 
                                self.robots[name].search_range, color='green', fill=False)
            plt.gca().add_patch(circle)
            plt.text(self.robots[name].pos[0], self.robots[name].pos[1], str(i), fontsize=16, color='black')

        for i in range(self.n_searcher):
            name = 'searcher_' + str(i)
            circle = plt.Circle((self.robots[name].pos[0], self.robots[name].pos[1]), 
                                self.robots[name].size, color='red', fill=True)
            plt.gca().add_patch(circle)
            circle = plt.Circle((self.robots[name].pos[0], self.robots[name].pos[1]), 
                                self.robots[name].search_range, color='red', fill=False)
            plt.gca().add_patch(circle)
            plt.text(self.robots[name].pos[0], self.robots[name].pos[1], str(i), fontsize=16, color='black')

        plt.gca().spines['right'].set_color('none')
        plt.gca().spines['left'].set_color('none')
        plt.gca().spines['top'].set_color('none')
        plt.gca().spines['bottom'].set_color('none')
        plt.axis('tight')
        plt.xlim(0, self.env_size[0])
        plt.ylim(0, self.env_size[1])
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        ax = plt.gca()
        ax.set_aspect(1)

        plt.draw()
        plt.pause(self.dt * 1)
        plt.clf()


if __name__ == '__main__':
    env_size = (8, 8)
    n_hider = 1
    n_searcher = 4
    dt = 0.1 
    max_step = 50
    hider_size = 0.2
    hider_search_range = 2.0
    hider_max_vel = 1.0
    searcher_size = 0.2
    searcher_search_range = 0.5
    searcher_max_vel = 0.7
    render_mode = 'human'
    queue_maxlen = 10
    queue_takelen = 6
    history_len = 5

    env = Environment(env_size, dt, render_mode, n_hider, n_searcher, 
                      max_step, queue_maxlen, queue_takelen, history_len,
                      hider_size, hider_search_range, hider_max_vel, 
                      searcher_size, searcher_search_range, searcher_max_vel)
    observations = env.reset()

    while not env.done and not env.truncated:
        # this is where you would insert your policy
        actions = {}
        for name in env.robots:
            cmd_angle = np.random.rand()
            actions[name] = [cmd_angle]

        observations, rewards, dones, truncated = env.step(actions)