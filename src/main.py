import gym
import random
import numpy as np

def EpisodeStartStr(num):
	epi_str = '============\n'
	epi_str += 'Episode ' + str(num) + '\n'
	epi_str += '============\n\n'
	return epi_str
try:
	gym = reload(gym)
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
	print 'Already registered'

env = gym.make('MyHex-v0')

for i_episode in range(1):
	print EpisodeStartStr(i_episode+1)

	initial_gamestate = env.reset()
	gamestate = initial_gamestate

	for i_step in range(3):
	    env.render()


    	''' Legal Actions '''
    	legal_actions = env.get_possible_actions(gamestate)

    	''' Random action '''
    	action = np.random.choice(legal_actions)

    	gamestate, reward, done, info = env.step(action)

    	#env.make_move(gamestate, action, 0)