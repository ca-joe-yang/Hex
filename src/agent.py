import numpy as np

def RandomAgent(HexEnv):
	gamestate = HexEnv.state
	
	legal_actions = HexEnv.get_possible_actions(gamestate)
	action = np.random.choice(legal_actions)

	return action