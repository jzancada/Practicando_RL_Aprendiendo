import numpy as np
import matplotlib.pyplot as plt 
from ddpg_torch import Agent
from utils import plot_learning_curve
from env import Env

np.random.seed(0)

if __name__ == '__main__':
    env = Env()
    load_checkpoint = True

    # epsilon a cero
    agent = Agent(alpha=0.0001, beta=0.001, 
                    input_dims=(11,), tau=0.001,
                    batch_size=64*4, fc1_dims=20, fc2_dims=15, 
                    n_actions=2)

    if load_checkpoint:
        agent.load_models()

    # simulacion
    observation, info = env.reset_fix_q_0(0)

    score = 0
    reward_v=[]
    action_v=[]
    precio_red_v=[]
    q_v = []

    for t in range(24*4):
        q_v.append(env.q)
        precio_red_v.append(observation[1])

        print('t = %3.1f' % env.t, ' q = %3.1f' % env.q)

        action = agent.choose_action(observation)
        action_v.append (action[0])

        observation_, reward, done, info = env.step(action, info)
        score += reward
        observation = observation_

        reward_v.append(reward)
        print('  accion=%d' % action[0] , ' reward = %6.1f' % reward, ' score = %6.1f' % score)



precio_red_v = np.array(precio_red_v)



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
plt.bar([k for k in range(len(q_v))],      q_v,        color = 'red',   label='q')
plt.bar(np.array([k for k in range(len(action_v))])+1, np.array(action_v)*1/4, color = 'green', alpha=0.5, label='a')
plt.plot((precio_red_v-precio_red_v.mean())/40*10,'.-', color='black', label='precio_red_v')
plt.legend()

plt.show(block=True)



