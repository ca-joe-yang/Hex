from gym.envs.board_game.hex import HexEnv
import numpy as np

def RandomAgent(gameState):
	
	legal_actions = HexEnv.get_possible_actions(gameState)
	if len(legal_actions) == 0:
		action = 'resign'
	else:
		action = np.random.choice(legal_actions)
	return action

def OnlyAttackAgent(gameState):
	action = 0


	return action

def evaluationFunction(gameState):
	reward = HexEnv.game_finished(gameState)
	return reward

def ExpectimaxAgent(gameState, max_depth=1):

	legal_actions = HexEnv.get_possible_actions(gameState)	
	if len(legal_actions) == 0:
		return 'resign'
	
	INT_MAX = 999999
	PLAYER = 0
	
	def getExpectimaxScoreAction(gameState, agentIndex, depth):
		legal_actions = HexEnv.get_possible_actions(gameState)	
		reward = HexEnv.game_finished(gameState)
		if depth == 0 or reward != 0:
			return reward, 0

		bestActions = []
		if agentIndex == PLAYER:
			bestScore = -INT_MAX
			for act in legal_actions:
				sucState = gameState.copy()
				HexEnv.make_move(sucState, act, agentIndex)
				score, a = getExpectimaxScoreAction(sucState, 1-PLAYER, depth)
				if score > bestScore:
					bestScore = score
					bestActions = []
				if score == bestScore:
					bestActions.append(act)
		
		elif agentIndex == 1-PLAYER:
			bestScore = 0.0
			for act in legal_actions:
				sucState = gameState.copy()
				HexEnv.make_move(sucState, act, agentIndex)
				score, a = getExpectimaxScoreAction(sucState, PLAYER, depth-1)
				bestScore += float(score)
			bestScore /= float(len(legal_actions))
		
		return bestScore, bestActions

	bestScore, bestActions = getExpectimaxScoreAction(gameState, PLAYER, max_depth)
		
	bestAction = np.random.choice(bestActions)
	#bestAction = bestActions[0]
	#print bestScore, bestAction
	return bestAction


