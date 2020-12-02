#
import numpy as np
import matplotlib.pyplot as plt 

np.random.seed(0)

class Env:
    a_space=[]
    def __init__(self):
        self.N_T = 24*2
        self.N_Q = 20
        self.N_A = 3
        self.a_space = np.array([-1,0,1])

    def reset(self):
        t=0
        q_space = np.arange(self.N_Q)
        q=np.random.choice(q_space)
        self.obs = [t, q]
        return self.obs

    def step(self, obs, a):
        done = False
        error = False
        reward = 0

        cost = np.ones(self.N_T)
        cost[14] = 2

        t = obs[0]
        q = obs[1]

        t_next = t + 1
        q_next = q

        if a ==  1:
            q_next = q_next + 1
        if a == -1:
            q_next = q_next - 1

        if q_next > self.N_Q-1:
            q_next =  self.N_Q-1
            error = True

        if q_next < 0:
            q_next = 0
            error = True

        delta_q = q_next - q
        reward = -cost[ t % 24 ]*delta_q
        # consumo bat
        reward -= np.abs(delta_q)*0.01

        if error or t_next == self.N_T:
            done = True

        self.obs = [t_next, q_next] 
        return self.obs, reward, done

class Actor:
    Q={}
    V={}
    def __init__(self, env):
        self.N_S = env.N_T * env.N_Q
        for s in range (self.N_S):
            for a in env.a_space:
                self.Q[s,a]=0
        self.env = env

    def to_s(self, obs):
        t = obs[0]
        q = obs[1]
        s = t*self.env.N_Q + q
        return s

    def get_action(self, obs):
        EPSILON = 0.1
        s = self.to_s(obs)
        actions = [self.Q[s,a] for a in self.env.a_space]
        a_ind = np.argmax(actions)
        a = self.env.a_space[a_ind]
        p = np.random.uniform(0,1)
        if p < EPSILON:
            a = np.random.choice(self.env.a_space)
        return a

    def update(self, obs, a, obs_next, reward):
        t = obs[0]
        q = obs[1]
        s = t*self.env.N_Q + q
        t_next = obs_next[0]
        q_next = obs_next[1]
        s_next = t_next*self.env.N_Q + q_next
        ALPHA = 0.05
        GAMMA = 0.999
        Q_next = [self.Q[s_next,a] for a in self.env.a_space]
        V_s_next = np.max(Q_next)
        delta = reward + GAMMA*V_s_next - self.Q[s,a]
        self.Q[s,a] = self.Q[s,a] + ALPHA * delta
        return delta

    def print_V(self):
        for q in range(self.env.N_Q):
            str = '{:3d}'.format(q)
            for t in range(self.env.N_T):
                s = t*self.env.N_Q + q
                str += '{:9.2f}'.format(self.V[s])
            print (str)

    def get_V(self):
        total = 0
        for q in range(self.env.N_Q):
            for t in range(self.env.N_T):
                s = t*self.env.N_Q + q
                Q = [self.Q[s,a] for a in self.env.a_space]
                self.V[s] = np.max(Q)
                total += self.V[s]
        return total

if __name__ == "__main__":
    env = Env()
    act = Actor(env)

    N_episodes = 300
    total_v =[]
    delta_v = []
    delta_v_episode = []

    for i in range(N_episodes):
        delta_v=[]
        obs = env.reset()

        done = False
        while not done:
            for t in range (env.N_T-1):
                for q in range (env.N_Q):
                    obs = [t, q]
                    a = act.get_action(obs)
                    obs_next, reward, done = env.step(obs, a)
                    delta = act.update(obs, a, obs_next, reward)
                    obs = obs_next
                    delta_v.append(delta)
        total = act.get_V()

        total_v.append(total)
        delta_v_episode.append(np.max(delta_v))

        if i % 1 == 0:
            print('iter %d' % i , ' total_v %.1f' % total)
            act.print_V()

            plt.figure(1)
            plt.plot(total_v)
            plt.draw()
            plt.show(block=False)

            plt.figure(2)
            plt.plot(delta_v_episode)
            plt.draw()
            plt.show(block=False)
            plt.pause(0.01)

    act.print_V()

    plt.figure(1)
    plt.plot(total_v)
    plt.draw()
    plt.show(block=False)

    plt.figure(2)
    plt.plot(delta_v_episode)
    plt.draw()
    plt.show(block=False)
    plt.pause(0.01)

# the end
plt.show(block=True)

plt.pause()