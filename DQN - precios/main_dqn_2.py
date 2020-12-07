import numpy as np
import matplotlib.pyplot as plt 
from dqn_agent import DQNAgent
from utils import plot_learning_curve
from env import Env

np.random.seed(0)

if __name__ == '__main__':
    env = Env()
    load_checkpoint = True

    # epsilon a cero
    agent = DQNAgent(gamma=0.99, epsilon=0, lr=0.0002,
                     input_dims=(env.N_Q+env.N_T,),
                     n_actions=3, mem_size=50000, eps_min=0,
                     batch_size=32, replace=1000, eps_dec=1e-6,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='car_env')

    if load_checkpoint:
        agent.load_models()

    # simulacion
    observation = env.reset_fix(0, 0)

    score = 0
    reward_v=[]
    q_v = []
    for t in range(24*10):

        q_v.append(env.q)
        print('t = %3.1f' % env.t, ' q = %3.1f' % env.q)

        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        observation = observation_

        reward_v.append(reward)
        print('  accion=%d' % action , ' reward = %6.1f' % reward, ' score = %6.1f' % score)

plt.figure()
plt.plot(reward_v,'.-', label='reward')

plt.figure()
plt.plot(q_v,'.-',label='q')
plt.show(block=True)



