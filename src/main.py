import gym
import random
import numpy as np
import agent

def EpisodeStartStr(num):
	epi_str = '============\n'
	epi_str += 'Episode ' + str(num) + '\n'
	epi_str += '============\n\n'
	return epi_str
try:
	gym.envs.register(
    	id='MyHex-v0',
    	entry_point='gym.envs.board_game:HexEnv',
    	kwargs={
        	'player_color': 'black',
        	'opponent': 'random',
        	'observation_type': 'numpy3c',
       		'illegal_move_mode': 'lose',
        	'board_size': 5,
    	},
	)
except Exception as e:
	print('Already registered')

env = gym.make('MyHex-v0')

for i_episode in range(1):
	print(EpisodeStartStr(i_episode+1))

	initial_gamestate = env.reset()
	gamestate = initial_gamestate
	env.render()

	while not env.done:
		action = agent.RandomAgent(env)

		gamestate, reward, done, info = env.step(action)

		env.render()
    	#env.make_move(gamestate, action, 0)
