import gym
import random
import numpy as np
import agent
from imp import reload 

reload(gym)
reload(agent)

def EpisodeStartStr(num):
	epi_str = '============\n'
	epi_str += 'Episode ' + str(num) + '\n'
	epi_str += '============\n\n'
	return epi_str
try:
	gym.envs.register(
    	id='MyHex5x5-v0',
    	entry_point='gym.envs.board_game:HexEnv',
    	kwargs={
        	'player_color': 'black',
        	'opponent': agent.RandomAgent,
        	'observation_type': 'numpy3c',
       		'illegal_move_mode': 'lose',
        	'board_size': 5,
    	},
	)
except Exception as e:
	print('Already registered')

env = gym.make('MyHex5x5-v0')

'''
	BLACK = 0
	WHITE = 1
'''

EPISODE_NUM = 100
VERBOSE = False

win_count = 0.0
for i_episode in range(EPISODE_NUM):

	initial_gamestate = env.reset()
	gamestate = initial_gamestate
	
	if VERBOSE:
		print(EpisodeStartStr(i_episode+1))
		env.render()

	while len(env.get_possible_actions(env.state)) != 0:
		#action = agent.RandomAgent(env.state)
		action = agent.ExpectimaxAgent(env.state, 1)

		gamestate, reward, done, info = env.step(action)
		

		if done:
			break

		if VERBOSE:
			env.render()

	if reward == 1:
		if VERBOSE:
			print('Win')
		win_count += 1
	else:
		if VERBOSE:
			print('Lose')

print('Win Rate: ', win_count / EPISODE_NUM)