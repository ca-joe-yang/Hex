import numpy as np
from __future__ import division

import time
from math import log, sqrt
from random import choice

class Agent:
	def __init__(self):
		return

	@abstractmethod
	def getAction(self, gameState): 
		pass

class RandomAgent(Agent):

	def __init__(self):
		return

	def getAction(self, gameState):
		legalActions = gameState.getLegalActions()
		actionIndex = np.random.choice(range(len(legalActions)))
		action = legalActions[actionIndex]
		return action


class HumanAgent(Agent):
	
	def __init__(self):
		return

	def getAction(self, gameState)
		legalActions = gameState.getLegalActions()
		while True:
			yourMoveStr = input('Your move: ')
			tokens = yourMoveStr.split(',')
			if len(tokens) != 2:
				print('Illegal input!')
				continue
			action = tuple([int(i) for i in tokens])
			if action not in legalActions:
				print('Illegal action!')
				continue
			break
		return action


class UCTNode(object):
	__slots__ = ('value', 'visits')

	def __init__(self, value=0, visits=0):
		self.value = value
		self.visits = visits

class UCTAgent(Agent):

	def __ init__(self, **kwargs):
		self.nodes = {}

		self.calculationTime = float(kwargs.get('time', 30))
		self.maxActions = int(kwargs.get('max_actions', 1000))

		# Exploration constant, increase for more exploratory actions,
		# decrease to prefer actions with known higher win rates.
		self.C = float(kwargs.get('C', 1.4))

	def getAction(self, gameState):
		# Causes the AI to calculate the best action from the
		# current game state and return it.

		self.max_depth = 0
		#self.data = {}
		self.nodes.clear()

		nowPlayer = gameState.nextPlayer
		legalActions = gameState.getLegalActions()

		# Bail out early if there is no real choice to be made.
		if len(legalActions) == 0:
			return None
		if len(legalActions) == 1:
			return legalActions[0]

		simulationCount = 0
		beginTime = time.time()
		while time.time() - beginTime < self.calculationTime:
			self.runSimulation()
			simulationCount += 1

		print('Simulation counts: ', simulationCount)
		print('Search max depth: ', self.maxDepth)
		print('Time elapsed: ', time.time() - beginTime)



		# Store and display the stats for each possible action.
		#self.data['actions'] = self.calculate_action_values(state, player, legal)
		#for m in self.data['actions']:
		#	print self.action_template.format(**m)

		# Pick the action with the highest average value.
		#return self.board.unpack_action(self.data['actions'][0]['action'])


	def runSimulation(self, gameState):

		# Plays out a "random" game from the current position,
		# then updates the statistics tables with the result.

		# A bit of an optimization here, so we have a local
		# variable lookup instead of an attribute access each loop.
		
		nodes = self.nodes

		explored = set()
		nowPlayer = gameState.nextPlayer
		maxActions = self.maxActions

		expand = True
		for t in range(maxActions):
			legalActions = gameState.getLegalActions()
			nextStates = [gameState.getNextState(action) for action in legalActions]

			if all( (player, nextState) in nodes for nextState in nextStates):
				# UCB1
				logSum = log( sum(nodes[ (player, nextState) ].visits for nextState in nextStates) or 1 )
				value, state = max(
					(( nodes[ (player, nextState) ].value / (nodes[ (player, nextState) ].visits or 1) ) +
					self.C * sqrt( logSum / (nodes[ (player, nextState) ].visits or 1)), nextState )
					for nextState in nextStates
				)
			else:
				state = choice(nextStates)

			if expand and (player, state) not in nodes:
				expand = False
				nodes[ (player, state) ] = UCTNode()
				if t > self.maxDepth:
					self.maxDepth = t

			explored.add((player, state))
			if state.isGoalState():
				break

		# Back-propagation
		winner = state.getWinner()
		reward = {}
		for player in [1,2]:
			if winner == 0:
				reward[player] = 0.0
			elif winner == player:
				reward[player] = 1.0
			else:
				reward[player] = -1.0

		for player, state in explored:
			if (player, state) not in nodes:
				continue
			S = nodes[ (player, state) ]
			S.visits += 1
			S.value += reward[player]


def NoDeadCellRandomAgent(gameState):
	legalActions = gameState.getLegalActions()
	goodActions = [a for a in legalActions if not gameState.isDeadCell(a)]
	actionIndex = np.random.choice(range(len(goodActions)))
	action = goodActions[actionIndex]
	return action

def BetterRandomAgent(gameState):
	action = gameState.mustPlayAction()
	if action != None:
		return action
	return NoDeadCellRandomAgent(gameState)
















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

def MonteCarloTreeSearchAgent(gameState):
	from mcts import UCTValues
	uct = UCTValues(gameState)
	uct.history.append(gameState)
	return uct.get_action()
