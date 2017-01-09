from __future__ import division
import numpy as np
from abc import ABCMeta, abstractmethod

import time
from math import log, sqrt
import random

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

	def getAction(self, gameState):
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

	def __init__(self, value=0.0, visits=0):
		self.value = value
		self.visits = visits

	def __str__(self):
		return('value: ' + str(self.value) + '\nvisits: ' + str(self.visits) + '\n')


class UCTAgent(Agent):

	def __init__(self, **kwargs):
		self.nodes = {}

		self.simulationTimeLimit = float(kwargs.get('time', 30))
		self.simulationActionsLimit = int(kwargs.get('max_actions', 1000))

		# Exploration constant, increase for more exploratory actions,
		# decrease to prefer actions with known higher win rates.
		self.C = float(kwargs.get('C', 1.4))

	def getAction(self, gameState):
		# Causes the AI to calculate the best action from the
		# current game state and return it.

		self.maxDepth = 0
		#self.data = {}
		self.nodes.clear()

		legalActions = gameState.getLegalActions()

		# Bail out early if there is no real choice to be made.
		if len(legalActions) == 0:
			return None
		if len(legalActions) >= 25:
			return random.choice(legalActions)

		simulationCount = 0
		beginTime = time.time()
		while time.time() - beginTime < self.simulationTimeLimit:
			self.runSimulation(gameState)
			simulationCount += 1

		print('Simulation counts: ', simulationCount)
		print('Search max depth: ', self.maxDepth)
		print('Time elapsed: ', time.time() - beginTime)
		#print(self.nodes.keys()[1])

		player = gameState.nextPlayer
		bestAction = None
		bestScore = -9999
		for action in legalActions:
			nextState = gameState.getNextState(action, player)
			if (player, nextState) not in self.nodes:
				continue
			value = self.nodes[ (player, nextState) ].value
			visits = self.nodes[ (player, nextState) ].visits
			score = value/visits
			if score > bestScore:
				bestScore = score
				bestAction = action

		return bestAction

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
		currentState = gameState.copy()

		expand = True
		for t in range(self.simulationActionsLimit):
			#print(t)
			currentPlayer = currentState.nextPlayer
			legalActions = currentState.getLegalActions()
			nextStates = [currentState.getNextState(action) for action in legalActions]

			if all( (currentPlayer, nextState) in nodes for nextState in nextStates):
				# UCB1
				logSum = log( sum(nodes[ (currentPlayer, nextState) ].visits for nextState in nextStates) or 1 )
				value, currentState = max(
					(( nodes[ (currentPlayer, nextState) ].value / (nodes[ (currentPlayer, nextState) ].visits or 1) ) +
					self.C * sqrt( logSum / (nodes[ (currentPlayer, nextState) ].visits or 1)), nextState )
					for nextState in nextStates
				)
			else:
				currentState = random.choice(nextStates)

			if expand and (currentPlayer, currentState) not in nodes:
				expand = False
				nodes[ (currentPlayer, currentState) ] = UCTNode()
				if t > self.maxDepth:
					self.maxDepth = t

			explored.add((currentPlayer, currentState))
			#print(state)
			if currentState.isGoalState():
				break
		#print(currentState)
		# Back-propagation
		for player, state in explored:
			if (player, state) not in nodes:
				continue
			S = nodes[ (player, state) ]
			#print(S)
			S.visits += 1
			#print(state.getReward(player))
			S.value += currentState.getReward(player)
			#print(self.nodes[(player, state)])


def NoDeadCellRandomAgent(gameState):
	legalActions = gameState.getLegalActions()
	goodActions = [a for a in legalActions if not gameState.isDeadCell(a)]
	actionIndex = np.random.choice(range(len(goodActions)))
	action = goodActions[actionIndex]
	return action

class BetterRandomAgent(Agent):

	def __init__(self):
		return

	def getAction(self, gameState):
		action = gameState.mustPlayAction()
		if action != None:
			return action
		legalActions = gameState.getLegalActions()
		goodActions = [a for a in legalActions if not gameState.isDeadCell(a)]
		action = random.choice(goodActions)
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

def MonteCarloTreeSearchAgent(gameState):
	from mcts import UCTValues
	uct = UCTValues(gameState)
	uct.history.append(gameState)
	return uct.get_action()
