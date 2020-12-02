#
import numpy as np
import matplotlib.pyplot as plt 
import pickle

np.random.seed(0)

class Env:
    a_space=[]
    def __init__(self):
        self.N_T = 24*6
        self.N_Q = 20
        self.N_A = 3
        self.a_space = np.array([-2,0,2])

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
        cost[15] = 2
        cost[20] = 0
        cost[21] = 0
        cost[22] = 0
        cost[23] = 0

        t = obs[0]
        q = obs[1]

        t_next = t + 1
        q_next = q

        if a ==  2:
            q_next = q_next + 2
        if a == -2:
            q_next = q_next - 2

        #########################
        # algo de estadistica
        if (t_next % 24) == 10:
            q_next = q_next
        #########################

        if q_next > self.N_Q-1:
            q_next =  self.N_Q-1
            error = True

        if q_next < 0:
            q_next = 0
            error = True

        delta_q = q_next - q
        reward = -cost[ t % 24 ]*delta_q
        # consumo bat
        reward -= np.abs(delta_q)*0.1
        # anxiety
        reward += np.abs(delta_q)*0.05

        if error:
            reward -= 10

        if error or t_next == self.N_T:
            done = True

        self.obs = [t_next, q_next] 
        return self.obs, reward, done

    def init_obs(self, obs):
        self.obs = obs

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

    def get_action(self, obs, epsilon=0):
        s = self.to_s(obs)
        actions = [self.Q[s,a] for a in self.env.a_space]
        a_ind = np.argmax(actions)
        a = self.env.a_space[a_ind]
        p = np.random.uniform(0,1)
        if p < epsilon:
            a = np.random.choice(self.env.a_space)
        return a

    def update(self, obs, a, obs_next, reward):
        t = obs[0]
        q = obs[1]
        s = t*self.env.N_Q + q
        t_next = obs_next[0]
        q_next = obs_next[1]
        s_next = t_next*self.env.N_Q + q_next
        ALPHA = 0.1
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

def train(N_episodes, epsilon):
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
                    a = act.get_action(obs, epsilon)
                    obs_next, reward, done = env.step(obs, a)
                    delta = act.update(obs, a, obs_next, reward)
                    obs = obs_next
                    delta_v.append(delta)
        total = act.get_V()

        total_v.append(total)
        delta_v_episode.append(np.max(delta_v))

        if i % 10 == 0:
            print('iter %d' % i , ' total_v %.1f' % total)
            act.print_V()

            plt.figure(1)
            plt.plot(total_v)
            plt.grid()
            plt.draw()
            plt.show(block=False)

            plt.figure(2)
            plt.plot(delta_v_episode)
            plt.grid()
            plt.draw()
            plt.show(block=False)
            plt.pause(0.01)

##############################################
if __name__ == "__main__":
    env = Env()
    act = Actor(env)

    if False: # true entrena
        epsilon = 0.1
        train(N_episodes = 3_00, epsilon = epsilon)
        print('Fin Train')
        # guardo en disco 

        output = open('config.pickle', 'wb')
        pickle.dump(act.Q, output, pickle.HIGHEST_PROTOCOL)
        output.close()

    else:
        input = open('config.pickle', 'rb')
        act.Q = pickle.load(input)
        input.close()

    # ejemplo de un recorrido
    t=0
    q=19
    obs = np.array([t,q])
    env.init_obs(obs)

    q_v = []
    R_v = []
    r_v = []
    R = 0
    done = False
    while not done:
        env.init_obs(obs)
        a = act.get_action(obs, epsilon=0  )
        s = act.to_s(obs)
        obs_next, reward, done = env.step(obs, a)
        print('t=%.1f'%obs[0] , ' q=%.1f'%obs[1], ' a=%d'%a , 'r=%.1f', reward)
        R = R + reward
        q_v.append(obs[1])
        R_v.append(R)
        r_v.append(reward)
        obs = obs_next

    plt.figure()
    plt.plot(q_v,'o-', color='red')
    plt.grid(True)
    plt.legend('q')
    plt.show(block=False)

    plt.figure()
    plt.plot(R_v,'o-', label='return', color='blue')
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(r_v,'o-', label='reward', color='black')
    plt.grid(True)
    plt.legend()
    plt.show(block=True)

    plt.show(block=True)
