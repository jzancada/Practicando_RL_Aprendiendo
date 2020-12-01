import gym  

env = gym.make('FrozenLake-v0')
observation =  env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action) # take a random action
    print('action ', action)
    print('observation ', observation)
    print('done ', done)
env.close()