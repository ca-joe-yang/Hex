import gym

env = gym.make('Hex9x9-v0')
env.reset()

for i in range(1000):

    env.render()
