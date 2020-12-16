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
    agent = DQNAgent(gamma=0.99, 
        epsilon=0, lr=0.002,
                     input_dims=(25,), # (q, c)
                     n_actions=5, mem_size=50000, eps_min=0,
                     batch_size=64, replace=1000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='car_env')

    if load_checkpoint:
        agent.load_models()

    # network
    print(agent.q_eval.fc2)
    print(agent.q_eval.fc2.weight)
    print(agent.q_eval.fc2.bias)

    # simulacion
    observation = env.reset_fix_q_0(0)

    score = 0
    reward_v=[]
    action_v=[]
    precio_red_v=[]
    q_v = []

    for t in range(24*2):
        q_v.append(env.q)
        precio_red_v.append(observation[1])

        print('t = %3.1f' % env.t, ' q = %3.1f' % env.q)

        action = agent.choose_action(observation)
        action_v.append (action)

        observation_, reward, done, info = env.step(action)
        score += reward
        observation = observation_

        reward_v.append(reward)
        print('  accion=%d' % action , ' reward = %6.1f' % reward, ' score = %6.1f' % score)

plt.figure()
plt.plot(reward_v,'.-', label='reward')
plt.legend()

plt.figure()
plt.plot(precio_red_v,'.-', color='black', label='precio_red_v')
#plt.plot(np.array(reward_v) / 6,'.-', color='red', label='reward')
for k in range(len(reward_v)):
    if action_v[k] ==0 or action_v[k] ==1:
       plt.plot(k, precio_red_v[k], 'o', color='red')
    if action_v[k] ==2:
       plt.plot(k, precio_red_v[k], 'o', color='blue')
    if action_v[k] ==3 or action_v[k] ==4:
       plt.plot(k, precio_red_v[k], 'o', color='green')

plt.legend()

plt.figure()
plt.plot(q_v,'.-',label='q')
plt.legend()

plt.figure()
plt.plot(action_v,'.-',label='a')
plt.legend()
plt.show(block=True)



