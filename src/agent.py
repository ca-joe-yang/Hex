import time
import random
import numpy as np
from abc import ABCMeta, abstractmethod
from math import log, sqrt
from collections import defaultdict
#from hex import HexState

class Agent:
	def __init__(self, player):
		self.player = player
		self.opponent = 3-player
		self.actionNum = 0
		self.capturedZone = []

	@abstractmethod
	def getAction(self, gameState): 
		pass

	def getName(self):
		if self.player == 1:
			return 'BLACK'
		elif self.player == 2:
			return 'WHITE'

	def evaluationFunction(self, gameState):
		begin = time.time()
		analysis = gameState.analysis()
		player = self.player
		opponent = self.opponent
		#print(time.time()-begin)

		score = 100 * gameState.getReward(player)
		score += 50 * len(analysis['check'][player])
		score -= 49 * len(analysis['check'][opponent])
		score -= 5 * len(analysis['dead'][player])
		score += 5 * len(analysis['dead'][opponent])
		score += 5 * len(analysis['captured'][player])
		score -= 5 * len(analysis['captured'][opponent])
		#print(time.time()-begin)
		return score


class RandomAgent(Agent):

	def __init__(self, player):
		super(RandomAgent, self).__init__(player)

	def getAction(self, gameState):
		action = random.choice(gameState.getLegalActions())
		return action

class BetterRandomAgent(Agent):

	def __init__(self, player):
		super(BetterRandomAgent, self).__init__(player)

	def getAction(self, gameState):
		#self.evaluateActions(gameState)
		action = random.choice(gameState.getGoodActions())
		return action		
		#capturedPairs = gameState.getAllCapturedPairs()
		#print(capturedPairs)

		#actions = [action for action in goodActions if not gameState.isCaptured(action)]

		'''
		lastAction = gameState.lastAction
		print('My: ', self.capturedZone)
		counterAction = self.isCaptured(lastAction)
		self.capturedZone = gameState.getCapturedZone()
		if counterAction:
			return counterAction
		actions = [action for action in goodActions if not self.isCaptured(action)]
		#print(actions)
		'''
		#return random.choice(goodActions)

class HumanAgent(Agent):
	
	def __init__(self, player):
		super(HumanAgent, self).__init__(player)
		self.betterRandomAgent = BetterRandomAgent(player)

	def getAction(self, gameState):
		#gameState.analysis()
		print(self.evaluationFunction(gameState))
		legalActions = gameState.getLegalActions()
		while True:
			#print(gameState.getCapturedZone(1))
			yourMoveStr = input('Your(' + self.getName() + ') move: ')
			if yourMoveStr == 'random' or yourMoveStr == 'r':
				return self.betterRandomAgent.getAction(gameState)
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


class MinimaxSearchAgent(Agent):

	def __init__(self, player):
		super(MinimaxSearchAgent, self).__init__(player)
	
	def getAction(self, gameState):
		return max( [action for action in gameState.getLegalActions()], 
			key=lambda x: self.evaluationFunction(gameState.getNextState(x)) )


class MyAgent(Agent):

	def __init__(self, player):
		super(MyAgent, self).__init__(player)
		self.randomAgent = RandomAgent(player)
		self.betterRandomAgent = BetterRandomAgent(player)
		self.monteCarloSearchAgent = MonteCarloSearchAgent(player)

	def getAction(self, gameState):
		if gameState.getAlreadyPlayedActionsNum() < 10:
			return self.randomAgent.getAction(gameState)
		else:
			return self.monteCarloSearchAgent.getAction(gameState)


class MonteCarloNode(object):
	__slots__ = ('value', 'visits')

	def __init__(self, value=0.0, visits=0):
		self.value = value
		self.visits = visits

	def __str__(self):
		return('{ value: ' + str(self.value) + ', visits: ' + str(self.visits) + '}')

	def update(self, reward):
		self.visits += 1
		self.value += reward

	def getScore(self):
		return self.value / (self.visits or 1)

class MonteCarloSearchAgent(Agent):

	def __init__(self, player, **kwargs):
		super(MonteCarloSearchAgent, self).__init__(player)
		self.simulationTimeLimit = float(kwargs.get('time', 30))
		# self.simulationActionsLimit = int(kwargs.get('max_actions', 1000))
		# Exploration constant, increase for more exploratory actions,
		# decrease to prefer actions with known higher win rates.
		self.C = float(kwargs.get('C', 1.4))
		self.tree = {} #defaultdict(MonteCarloNode)

	def getAction(self, gameState):
		# Causes the AI to calculate the best action from the
		# current game state and return it.

		assert not gameState.isGoalState()

		self.maxDepth = 0
		self.tree.clear()

		simulationCount = 0
		beginTime = time.time()
		while time.time() - beginTime < self.simulationTimeLimit:
			simulationCount += 1
			print('Simulation', simulationCount, end=' ')
			reward = self.runSimulation(gameState)
			print(', reward:', reward)

		print('Simulation counts:', simulationCount)
		print('Search max depth:', self.maxDepth)
		print('Time elapsed:', time.time() - beginTime)

		player = gameState.nextPlayer

		bestAction = max( [ action for action in gameState.getGoodActions() if gameState.getNextState(action) in self.tree ], 
				key=lambda x: self.tree[gameState.getNextState(x)].getScore())

		print('Average reward:', self.tree[gameState.getNextState(bestAction)].getScore())

		return bestAction

	def runSimulation(self, gameState):

		# Plays out a "random" game from the current position,
		# then updates the statistics tables with the result.

		# A bit of an optimization here, so we have a local
		# variable lookup instead of an attribute access each loop.
		
		explored = set()
		currentState = gameState.copy()
		player = currentState.nextPlayer

		expand = True
		depth = 0

		while True:
			depth += 1
			nextStates = [currentState.getNextState(action) for action in currentState.getGoodActions()]

			if all( nextState in self.tree for nextState in nextStates):
				# UCB1
				logSum = log( sum( self.tree[ nextState ].visits for nextState in nextStates ) or 1 )
				currentState = max( [ nextState for nextState in nextStates], 
					key=lambda x: self.tree[x].getScore() + self.C * sqrt(logSum / (self.tree[x].visits or 1)) )
			else:
				currentState = random.choice(nextStates)

			if expand and currentState not in self.tree:
				expand = False
				self.tree[ currentState ] = MonteCarloNode()
				self.maxDepth = max(depth, self.maxDepth)
			explored.add(currentState)
			if currentState.isGoalState():
				break

		#print(currentState)
		# Back-propagation
		reward = currentState.getReward(player)
		for state in explored:
			if state not in self.tree:
				continue
			self.tree[state].update(reward)
		return reward



		























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




