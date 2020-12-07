import argparse, os
import numpy as np
from dqn_agent import DQNAgent
from utils import plot_learning_curve

from env import Env

np.random.seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Vehiculos Electricos')
    # the hyphen makes the argument optional
    parser.add_argument('-n_epochs', type=int, default=10,
        help='Number of Epochs')
    args = parser.parse_args()

    env = Env()
    best_score = -np.inf
    load_checkpoint = False
    n_games = args.n_epochs

    agent = DQNAgent(gamma=0.99, 
        epsilon=1, lr=0.0002,
                     input_dims=2, # (q, c)
                     n_actions=3, mem_size=50000, eps_min=0.1,
                     batch_size=64, replace=1000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='car_env')

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    # if you want to record video of your agent playing, do a mkdir tmp && mkdir tmp/dqn-video
    # and uncomment the following 2 lines.
    #env = wrappers.Monitor(env, "tmp/dqn-video",
    #                    video_callable=lambda episode_id: True, force=True)
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action,
                                     reward, observation_, done)
                agent.learn()
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i,'score: ', score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.4f' % agent.epsilon, 'steps', n_steps)

        eps_history.append(agent.epsilon)

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

            x = [i+1 for i in range(len(scores))]
            plot_learning_curve(steps_array, scores, eps_history, figure_file)

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
