import gym  
from gym import spaces
import numpy as np

class Actor():
    alpha = 0.1
    gamma = 0.99
    bins_x=[]
    bins_x_dot=[]
    bins_theta=[]
    bins_theta_dot=[]
    n_x = 0
    n_x_dot = 0
    n_theta = 0
    n_theta_dot = 0
    obs=[]
    s=0
    V=[]

    def obs_to_s(self, obs):
        i_x         = np.digitize(obs[0], self.bins_x)
        i_x_dot     = np.digitize(obs[1], self.bins_x_dot)
        i_theta     = np.digitize(obs[2], self.bins_theta)
        i_theta_dot = np.digitize(obs[3], self.bins_theta_dot)
        return s

    def __init__(self, obs_space, act_space):
        # theta
        x_max = obs_space.high
        x_min = obs_space.low
        self.bins_x             = np.linspace(x_min[0], x_max[0], 10)
        self.bins_x_dot         = np.linspace(x_min[1], x_max[1], 10)
        self.bins_theta         = np.linspace(x_min[2], x_max[2], 10)
        self.bins_theta_dot     = np.linspace(x_min[3], x_max[3], 10)
        self.n_x = len([self.bins_x, self.bins_x_dot])
        s = 1
        
    def reset(self, obs):
        self.obs = obs
        self.s = self.obs_to_s(obs)

    def get_action(self):
        x = self.obs[0]
        if x >0:
            a = 0
        else:
            a = 1
        return a

    def update(self, obs, reward):
        s_next = self.obs_to_s(obs)
        alpha = 0.1
        gamma = 0.99
        self.V[s] = self.V(s) + alpha*(reward + gamma*V(s_next) - V[s])
        self.obs = obs
        self.s = s_next
        pass


env = gym.make('CartPole-v1')

actor =Actor(env.observation_space, env.action_space)
n_episodes = 100
for i_episodes in range(n_episodes):
    if i_episodes % 100 == 0:
        print ('episode ', i_episodes)
    observation =  env.reset()
    actor.reset(observation)
    isDone = False
    while not isDone:
        a = actor.get_action()
        observation, reward, done, info = env.step(a) # take a random action
        actor.update(observation, reward)
        isDone = done
    env.close()


env.render()
