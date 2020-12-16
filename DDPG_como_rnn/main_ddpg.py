import numpy as np
import matplotlib.pyplot as plt
from ddpg_torch import Agent
from utils import plot_learning_curve
from env import Env

if __name__ == '__main__':
    env = Env()
    # agent = Agent(alpha=0.0001, beta=0.001, 
    #                 input_dims=(2,), tau=0.001,
    #                 batch_size=64, fc1_dims=40, fc2_dims=30, 
    #                 n_actions=1)
    agent = Agent(alpha=0.0001, beta=0.001, 
                    input_dims=(11+4,), tau=0.001,
                    batch_size=64*4, fc1_dims=20, fc2_dims=15, 
                    n_actions=2)
    n_games = 100
    filename = 'LunarLander_alpha_' + str(agent.alpha) + '_beta_' + \
                str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'

    # best_score = env.reward_range[0]
    best_score = 1
     
    score_history = []
    for i in range(n_games):
        observation, env_info = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        env_info['epoch']=i
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, env_info = env.step(action, env_info)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
            plt.figure(1)
            plt.plot(score_history)
            plt.show(block=False)
            plt.pause(0.001)

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)




