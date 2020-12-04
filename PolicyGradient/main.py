import matplotlib.pyplot as plt 
import numpy as np
from reinforce_torch import PolicyGradientAgent

class Env():
    def __init__(self):
        self.cost = [0,0,0,1000]
        self.q_space=np.array([0,1])
        self.obs = self.reset()
        self.info = []

    def reset(self):
        t = 0
        q = np.random.choice(self.q_space)
        self.obs = [t, q]
        return self.obs 

    def step(self, action):
        (t,q)=self.obs
        n = len(self.cost)
        penalizacion_q = 0
        done = False

        t_ = t +1

        if action == 0:
            q_ = q

        if action == 1:
            q_ = q + 1
            if (q_ > 1):
                q_ = 1
                penalizacion_q = -1e-9
        if action == 2:
            q_ = q - 1
            if (q_ < 0):
                q_ = 0
                penalizacion_q = -1e-9

        self.obs = [t_ , q_ ]

        delta_q = q_ - q
        reward = -self.cost[t]*delta_q + penalizacion_q

        if t_ >= n:
            done = True
        return  self.obs, reward, done, self.info

##############################################    
def plot_learning_curve(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.grid(True)
    plt.savefig(figure_file)

##############################################    
if __name__ == "__main__":
    env = Env()
    n_games = 6000
    agent = PolicyGradientAgent(gamma = 0.99, lr=0.0005, input_dims=[2], n_actions =3)
    fname = 'REINFORCE_' + 'env_lr' + str(agent.lr) + '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    scores=[]
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_reward(reward) 
            observation = observation_
        agent.learn()
        scores.append(score)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print('episode ', i, 'score %10.2f' % score,
                    'average score %10.2f' % avg_score)
            x = [i+1 for i in range(len(scores))]
            plot_learning_curve(scores, x, figure_file)
            plt.show(block=False)
            plt.pause(0.1)
        
    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(scores, x, figure_file)


print('main')