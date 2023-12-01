import numpy as np
import matplotlib.pyplot as plt

class Robot:
    def __init__(self, size, search_range, max_vel, 
                 position, env_size):
        self.size = size
        self.search_range = search_range
        self.max_vel = max_vel
        self.pos = position
        self.env_size = env_size

    def set_pos(self, position):
        self.pos = position

    def move(self, cmd_vel, cmd_angle, dt):
        # cmd range from 0 to 1
        desired_vel = cmd_vel * self.max_vel
        desired_angle = 2 * np.pi * cmd_angle
        vx = np.cos(desired_angle) * desired_vel
        vy = np.sin(desired_angle) * desired_vel
        self.pos[0] += vx * dt
        self.pos[1] += vy * dt
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

        self.no_vision_state = [-1, -1, -1, -1]
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
        if render_mode == 'human':
            plt.figure(figsize=(7, 7))

    def get_visible_tag(self, name):
        visible = {}
        distance = {}
        for bot in self.robots:
            if bot != name:
                dis = np.linalg.norm(x = [self.robots[name].pos[0] - self.robots[bot].pos[0], \
                                          self.robots[name].pos[1] - self.robots[bot].pos[1]], ord = 2)
                if dis < self.robots[name].search_range + self.robots[bot].size:
                    visible[bot] = True
                    distance[bot] = dis
                else:
                    visible[bot] = False
        return visible, distance
    
    def generate_obs(self, true_state):
        obs = {}
        for name in self.robots:
            obs[name] = []
            visible, distance = self.get_visible_tag(name)
            for i in range(self.n_hider):
                hider = 'hider_' + str(i)
                if name == hider:
                    obs[name].append(true_state[hider])
                else:
                    if visible[hider]:
                        obs[name].append(true_state[hider])
                    else:
                        obs[name].append(self.no_vision_state)
            for i in range(self.n_searcher):
                searcher = 'searcher_' + str(i)
                if name == searcher:
                    obs[name].append(true_state[searcher])
                else:
                    if visible[searcher]:
                        obs[name].append(true_state[searcher])
                    else:
                        obs[name].append(self.no_vision_state)
            obs[name] = np.array(obs[name]).flatten()
        return obs

    def reset(self):
        self.finish = False
        self.life_time = 0

        true_state = {}
        hider_x = np.random.rand(self.n_hider) * self.env_size[0]
        searcher_x = np.random.rand(self.n_searcher) * self.env_size[0]
        for i in range(self.n_hider):
            name = 'hider_' + str(i)
            self.robots[name].set_pos([hider_x[i], self.env_size[1]])
            true_state[name] = [self.robots[name].pos[0], self.robots[name].pos[1], \
                                self.env_size[0] - self.robots[name].pos[0], \
                                self.env_size[1] - self.robots[name].pos[1]]
        for i in range(self.n_searcher):
            name = 'searcher_' + str(i)
            self.robots[name].set_pos([searcher_x[i], 0])
            true_state[name] = [self.robots[name].pos[0], self.robots[name].pos[1], \
                                self.env_size[0] - self.robots[name].pos[0], \
                                self.env_size[1] - self.robots[name].pos[1]]
        return self.generate_obs(true_state)
    
    def step(self, actions):
        true_state = {}
        for name in self.hider_names:
            self.robots[name].move(actions[name][0], actions[name][1], self.dt)
            true_state[name] = [self.robots[name].pos[0], self.robots[name].pos[1], \
                                self.env_size[0] - self.robots[name].pos[0], \
                                self.env_size[1] - self.robots[name].pos[1]]
        for name in self.searcher_names:
            self.robots[name].move(actions[name][0], actions[name][1], self.dt)
            true_state[name] = [self.robots[name].pos[0], self.robots[name].pos[1], \
                                self.env_size[0] - self.robots[name].pos[0], \
                                self.env_size[1] - self.robots[name].pos[1]]
        obs = self.generate_obs(true_state)

        if self.render_mode == 'human':
            self.show_env()

        self.life_time += 1
        if self.life_time > self.max_step: # check if get max_step
            self.finish = True

        hider_been_capture = None
        searcher_success = None
        for hider in self.hider_names: # check if a hider been capture
            for searcher in self.searcher_names:
                dis = np.linalg.norm(x = [self.robots[hider].pos[0] - self.robots[searcher].pos[0], \
                                        self.robots[hider].pos[1] - self.robots[searcher].pos[1]], ord = 2)
                if dis < self.robots[hider].size + self.robots[searcher].size:
                    self.finish = True
                    hider_been_capture = hider
                    searcher_success = searcher
                    break
            if self.finish:
                break

        rewards = {}
        dones = {}
        for name in self.hider_names:
            dones[name] = self.finish
            rewards[name] = self.dt * (1+self.n_searcher)
            visible, distance = self.get_visible_tag(name)
            for searcher in self.searcher_names:
                if searcher in distance:
                    rewards[name] -= -(self.dt/self.diagonal)*distance[searcher] + self.dt
        for name in self.searcher_names:
            dones[name] = self.finish
            rewards[name] = -self.dt * (1+self.n_hider)
            visible, distance = self.get_visible_tag(name)
            for hider in self.hider_names:
                if hider in distance:
                    rewards[name] += -(self.dt/self.diagonal)*distance[hider] + self.dt
                    
        if self.n_hider > 1:
            for i in range(self.n_hider-1):
                for j in range(i+1, self.n_hider):
                    dis = np.linalg.norm(x = [self.robots[self.hider_names[i]].pos[0] - self.robots[self.hider_names[j]].pos[0], \
                                        self.robots[self.hider_names[i]].pos[1] - self.robots[self.hider_names[j]].pos[1]], ord = 2)
                    if dis < self.robots[self.hider_names[i]].size + self.robots[self.hider_names[i]].size:
                        rewards[self.hider_names[i]] -= 10
                        rewards[self.hider_names[j]] -= 10
        if self.n_searcher > 1:
            for i in range(self.n_searcher-1):
                for j in range(i+1, self.n_searcher):
                    dis = np.linalg.norm(x = [self.robots[self.searcher_names[i]].pos[0] - self.robots[self.searcher_names[j]].pos[0], \
                                        self.robots[self.searcher_names[i]].pos[1] - self.robots[self.searcher_names[j]].pos[1]], ord = 2)
                    if dis < self.robots[self.searcher_names[i]].size + self.robots[self.searcher_names[i]].size:
                        rewards[self.searcher_names[i]] -= 10
                        rewards[self.searcher_names[j]] -= 10
        
        if hider_been_capture and searcher_success:
            rewards[hider_been_capture] -= 30
            rewards[searcher_success] += 30

        return obs, rewards, dones
    
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
            cmd_vel = np.random.rand()
            cmd_angle = np.random.rand()
            actions[name] = (cmd_vel, cmd_angle)

        observations, rewards, dones = env.step(actions)