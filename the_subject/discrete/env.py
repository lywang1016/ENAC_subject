import numpy as np
import matplotlib.pyplot as plt

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
        if action == 0: # stay
            self.vel = [0, 0]
        if action == 1: # up
            self.pos[1] += self.max_vel * dt
            self.vel = [0, self.max_vel]
        if action == 2: # down
            self.pos[1] -= self.max_vel * dt
            self.vel = [0, -self.max_vel]
        if action == 3: # left
            self.pos[0] -= self.max_vel * dt
            self.vel = [-self.max_vel, 0]
        if action == 4: # right
            self.pos[0] += self.max_vel * dt
            self.vel = [self.max_vel, 0]
        self.pos[0] = np.clip(self.pos[0], 0, self.env_size[0])
        self.pos[1] = np.clip(self.pos[1], 0, self.env_size[1])

class Environment:
    def __init__(self, env_size, dt, render_mode, n_hider, n_searcher, max_step,
                 hider_size, hider_search_range, hider_max_vel,
                 searcher_size, searcher_search_range, searcher_max_vel):
        self.n_hider = n_hider
        self.n_searcher = n_searcher
        self.env_size = env_size
        self.dt = dt
        self.max_step = max_step 
        self.finish = False
        self.life_time = 0
        self.render_mode = render_mode

        self.diagonal = np.linalg.norm(x = [env_size[0], env_size[1]], ord = 2)

        hider_x = np.random.rand(n_hider) * env_size[0]
        searcher_x = np.random.rand(n_searcher) * env_size[0]

        self.robots = {}
        self.hider_names = []
        self.searcher_names = []
        for i in range(n_hider):
            name = 'hider_' + str(i)
            self.hider_names.append(name)
            self.robots[name] = Robot(hider_size, hider_search_range, hider_max_vel, 
                                      [hider_x[i], env_size[1]], env_size)
        for i in range(n_searcher):
            name = 'searcher_' + str(i)
            self.searcher_names.append(name)
            self.robots[name] = Robot(searcher_size, searcher_search_range, searcher_max_vel, 
                                      [searcher_x[i], 0], env_size)
        
        distances = self.hider_searcher_distance()
        self.total_dis_last = sum(distances)
        self.min_dis_last = min(distances)

        if render_mode == 'human':
            plt.figure(figsize=(7, 7))

    def reset(self, show=True):
        self.finish = False
        self.life_time = 0

        hider_x = np.random.rand(self.n_hider) * self.env_size[0]
        searcher_x = np.random.rand(self.n_searcher) * self.env_size[0]
        for i in range(self.n_hider):
            name = 'hider_' + str(i)
            self.robots[name].set_pos([hider_x[i], self.env_size[1]])
        for i in range(self.n_searcher):
            name = 'searcher_' + str(i)
            self.robots[name].set_pos([searcher_x[i], 0])

        distances = self.hider_searcher_distance()
        self.total_dis_last = sum(distances)
        self.min_dis_last = min(distances)

        if self.render_mode == 'human' and show:
            self.show_env()
        
        return self.generate_obs()
    
    def step(self, actions):
        for name in self.hider_names:
            self.robots[name].move(actions[name], self.dt)
        for name in self.searcher_names:
            self.robots[name].move(actions[name], self.dt)
        obs = self.generate_obs()

        if self.render_mode == 'human':
            self.show_env()

        self.life_time += 1
        if self.life_time > self.max_step: # check if get max_step
            self.finish = True

        for hider in self.hider_names: # check if a hider been capture
            for searcher in self.searcher_names:
                dis = self.distance_from_a_look_b(hider, searcher)
                if dis < self.robots[hider].size + self.robots[searcher].size:
                    self.finish = True
                    break
            if self.finish:
                break

        rewards = {}
        dones = {}
        
        for name in self.hider_names:
            dones[name] = self.finish
            rewards[name] = 1   # reward for life 1 step more
            dis_to_bound = min([self.robots[name].pos[0], self.robots[name].pos[1], \
                                self.env_size[0] - self.robots[name].pos[0], \
                                self.env_size[1] - self.robots[name].pos[1]])
            if dis_to_bound < self.robots[name].size:   # punish for close to the boundary
                rewards[name] -= (self.robots[name].size - dis_to_bound) / self.robots[name].size
        for name in self.searcher_names:
            dones[name] = self.finish
            rewards[name] = -1  # punish for give hider life 1 step more
            dis_to_bound = min([self.robots[name].pos[0], self.robots[name].pos[1], \
                                self.env_size[0] - self.robots[name].pos[0], \
                                self.env_size[1] - self.robots[name].pos[1]])
            if dis_to_bound < self.robots[name].size:   # punish for close to the boundary
                rewards[name] -= (self.robots[name].size - dis_to_bound) / self.robots[name].size

        if self.n_hider > 1:
            for i in range(self.n_hider-1):
                for j in range(i+1, self.n_hider):
                    dis = self.distance_from_a_look_b(self.hider_names[i], self.hider_names[j])
                    if dis < self.robots[self.hider_names[i]].size + self.robots[self.hider_names[j]].size:
                        rewards[self.hider_names[i]] -= 10      # punish for colision
                        rewards[self.hider_names[j]] -= 10
        if self.n_searcher > 1:
            for i in range(self.n_searcher-1):
                for j in range(i+1, self.n_searcher):
                    dis = self.distance_from_a_look_b(self.searcher_names[i], self.searcher_names[j])
                    if dis < self.robots[self.searcher_names[i]].size + self.robots[self.searcher_names[j]].size:
                        rewards[self.searcher_names[i]] -= 10   # punish for colision
                        rewards[self.searcher_names[j]] -= 10

        distances = self.hider_searcher_distance()
        total_dis = sum(distances)
        min_dis = min(distances)

        total_dis_change = total_dis - self.total_dis_last
        min_dis_change = min_dis - self.min_dis_last
        dis_metric = 3*min_dis_change + total_dis_change
        for name in self.hider_names:
            rewards[name] += dis_metric     # reward for get far from searcher
        for name in self.searcher_names:
            rewards[name] -= dis_metric     # punish for get far from hider
        
        self.total_dis_last = total_dis
        self.min_dis_last = min_dis

        return obs, rewards, dones
    
    def hider_searcher_distance(self):
        res = []
        for hider in self.hider_names:
            for searcher in self.searcher_names:
                temp = self.distance_from_a_look_b(hider, searcher)
                res.append(temp)
        return res
    
    def distance_from_a_look_b(self, a, b):
        dis = np.linalg.norm(x = [self.robots[a].pos[0] - self.robots[b].pos[0], \
                                self.robots[a].pos[1] - self.robots[b].pos[1]], ord = 2)
        if dis < self.robots[a].search_range + self.robots[b].size:
            return dis
        else:
            return self.diagonal+1

    def generate_obs_from_a_look_b(self, a, b):
        dis = self.distance_from_a_look_b(a, b)
        if dis < self.diagonal:
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

        for i in range(self.n_searcher):
            name = 'searcher_' + str(i)
            circle = plt.Circle((self.robots[name].pos[0], self.robots[name].pos[1]), 
                                self.robots[name].size, color='red', fill=True)
            plt.gca().add_patch(circle)
            circle = plt.Circle((self.robots[name].pos[0], self.robots[name].pos[1]), 
                                self.robots[name].search_range, color='red', fill=False)
            plt.gca().add_patch(circle)

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

    env = Environment(env_size, dt, render_mode, n_hider, n_searcher, max_step,
                      hider_size, hider_search_range, hider_max_vel, 
                      searcher_size, searcher_search_range, searcher_max_vel)
    
    observations = env.reset()

    while not env.finish:
        # this is where you would insert your policy
        actions = {}
        for name in env.robots:
            action = np.random.randint(5)
            actions[name] = (action)

        observations, rewards, dones = env.step(actions)