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
RESIGN_MOVE = env.board_size ** 2

'''
	BLACK = 0
	WHITE = 1
'''

EPISODE_NUM = 100
VERBOSE = False
MODE = 'human'
if MODE == 'human':
	VERBOSE = True

win_count = 0.0
for i_episode in range(EPISODE_NUM):

	gameState = env.reset()
	
	if VERBOSE:
		print(EpisodeStartStr(i_episode+1))
		env.render()

	if MODE == 'human':
		while len(env.get_possible_actions(env.state)) != 0:
			your_move_str = input('Your move: ')
			try:
				your_action = int(your_move_str)
				assert your_action <= RESIGN_MOVE
			except Exception as e:
				try:
					your_move_coord = your_move_str.split(',')
		
					assert len(your_move_coord) == 2
					your_move_coord = [int(i)-1 for i in your_move_coord]
					print(your_move_coord)
					your_action = env.coordinate_to_action(gameState, your_move_coord)
					print(your_action)
				except Exception as e:
					print('Illegal Input!')
					continue

			gameState, reward, done, info = env.step(your_action)
			env.render()
			if done:
				break

	elif MODE == 'ai':
		while len(env.get_possible_actions(env.state)) != 0:
			#action = agent.RandomAgent(env.state)
			action = agent.ExpectimaxAgent(env.state, 1)

			if action == 'resign':
				action = RESIGN_MOVE

			gameState, reward, done, info = env.step(action)
			if VERBOSE:
				env.render()

			if done:
				break


	if reward == 1:
		if VERBOSE:
			print('Win')
		win_count += 1
	else:
		if VERBOSE:
			print('Lose')

print('Win Rate: ', win_count / EPISODE_NUM)