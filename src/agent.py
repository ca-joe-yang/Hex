import time
import random
import numpy as np
from abc import ABCMeta, abstractmethod
from math import log, sqrt
from collections import defaultdict
import networkx as nx
from hex import HexPlayer
import itertools

class Agent:
	def __init__(self, player):
		self.player = player
		self.opponent = 3-player
		self.actionNum = 0
		self.my_move_history = []
		self.opponent_moves_history = []

	@abstractmethod
	def getAction(self, gameState): 
		pass

	def getName(self):
		if self.player == 1:
			return 'BLACK'
		elif self.player == 2:
			return 'WHITE'

	def getRandomLegalAction(self, gameState):
		bestActions = gameState.getLegalActions()
		bestAction = random.choice(bestActions)
		return bestAction

	def getRandomGoodAction(self, gameState):
		bestActions = gameState.getGoodActions()
		bestAction = random.choice(bestActions)
		return bestAction


	def getReflexActions(self, gameState):
		if gameState.lastAction in gameState.bridgePairs[self.player]:
			return gameState.bridgePairs[self.player][gameState.lastAction]
		return gameState.getGoodActions()	


	def evaluationFunction(self, gameState, player):
		'''
		analysis = gameState.analysis()
		opponent = 3-player

		score = 1000 * gameState.getReward(player)
		score += 500 * len(analysis['winning'][player])
		score -= 500 * len(analysis['winning'][opponent])
		score -= 100 * analysis['shortest'][player]
		score += 100 * analysis['shortest'][opponent]
		score -= 5 * len(analysis['dead'][player])
		score += 5 * len(analysis['dead'][opponent])
		score += 5 * len(analysis['captured'][player])
		score -= 5 * len(analysis['captured'][opponent])
		score += 10 * analysis['bridges'][player]
		score -= 10 * analysis['bridges'][opponent]
		'''
		#print(gameState, score)
		return 0.0

	def evaluationFunction(self, gameState, player, **kwargs):
		method = kwargs.pop('method', 'sspl')
		if method == 'analysis':
			return self.evaluationByAnalysis(gameState, player)
		elif method == 'sspl':
			return self.evaluationBySecondShortestPathLength(gameState, player)


	def evaluationByAnalysis(self, gameState, player):
		
		N = HexState.BOARD_SIZE
		#actionSet = set(self.getGoodActions())

		#blackWinningActions = HexState.getWinningActions(self.board, HexPlayer.BLACK)
		#whiteWinningActions = HexState.getWinningActions(self.board, HexPlayer.WHITE)

		#actionSet -= whiteWinningActions
		#actionSet -= blackWinningActions

		#blackCells = self.getAllCells(HexPlayer.BLACK)
		#whiteCells = self.getAllCells(HexPlayer.WHITE)
		
		#blackDeadCells = [cell for cell in gameState.dead if cell in gameState.getCells(player=HexPlayer.BLACK)]
		#whiteDeadCells = [cell for cell in gameState.dead if cell in gameState.getCells(player=HexPlayer.BLACK)]
		#deadActions = set( [action for action in self.dead if action in actionSet] )
		#actionSet -= deadActions
		#print(time.time()-begin)
		#blackCapturedActions = [action for action in HexState.getNeighbors(self.board, blackCells) if self.isCapturedByPlayer(action, HexPlayer.BLACK)]
		#actionSet -= blackCapturedActions
		#print(time.time()-begin)
		#whiteCapturedActions = set([action for action in HexState.getNeighbors(self.board, whiteCells) if self.isCapturedByPlayer(action, HexPlayer.WHITE)])
		#actionSet -= whiteCapturedActions
		#print(time.time()-begin)

		#blackVulnerableActions = [action for action in actionSet if self.isVulnerableToPlayer(action, HexPlayer.BLACK)]
		#whiteVulnerableActions = [action for action in actionSet if self.isVulnerableToPlayer(action, HexPlayer.WHITE)]

		'''
		blackGraph = self.shannonGraphs[HexPlayer.BLACK]
		blackShortestPath = nx.shortest_path_length(
			blackGraph, 
			source=(1, 0), 
			target=(1, N+1),
		)

		whiteGraph = self.shannonGraphs[HexPlayer.WHITE]
		whiteShortestPath = nx.shortest_path_length(
			whiteGraph, 
			source=(0, 1),
			target=(N+1, 1),
		)

		result = {
			'winning': {
				1: blackWinningActions,
				2: whiteWinningActions
			},
			'dead': {
				1: blackDeadCells,
				2: whiteDeadCells,
				0: deadActions
			},
			'captured': {
				1: blackCapturedActions,
				2: whiteCapturedActions
			},
			'bridges': {
				HexPlayer.BLACK: HexState.getPlayerBridgesNum(self.board, HexPlayer.BLACK),
				HexPlayer.WHITE: HexState.getPlayerBridgesNum(self.board, HexPlayer.WHITE) 
			},
			'shortest': {
				HexPlayer.BLACK: blackShortestPath,
				HexPlayer.WHITE: whiteShortestPath
			}
		}
		'''

		#print(result)
		'''
		print('Black Winning:', blackWinningActions)
		print('White Winning:', whiteWinningActions)
		print('Black Dead:', blackDeadCells)
		print('White Dead:', whiteDeadCells)
		print('Dead Actions:', deadActions)
		print('Black Captured: ', blackCapturedActions)
		print('White Captured: ', whiteCapturedActions)
		'''
		#return result

		return 0.0

	def evaluationBySecondShortestPathLength(self, gameState, player):
		N = gameState.N
		if gameState.isGoalState():
			return gameState.getReward(player)
		d = gameState.getShortestDistanceSum(player)
		#secondShortestDistance = min( distance.remove(min(distance)) )
		shortest = min(d.values())
		
		for x in d:
			if d[x] == shortest:
				del d[x]
				break
		return (N+1-min(d.values())) / N


class RandomAgent(Agent):

	def __init__(self, player):
		super(RandomAgent, self).__init__(player)

	def getAction(self, gameState):
		return self.getRandomLegalAction(gameState)

class BetterRandomAgent(Agent):

	def __init__(self, player):
		super(BetterRandomAgent, self).__init__(player)

	def getAction(self, gameState):
		return self.getRandomGoodAction(gameState)	


class ReflexAgent(Agent):

	def __init__(self, player):
		super(ReflexAgent, self).__init__(player)

	def getAction(self, gameState):
		return self.getReflexAction(gameState)	

class HumanAgent(Agent):
	
	def __init__(self, player):
		super(HumanAgent, self).__init__(player)

	def getAction(self, gameState):
		legalActions = gameState.getLegalActions()
		print(self.evaluationFunction(gameState, self.player))
		while True:
			yourMoveStr = input('Your(' + self.getName() + ') move: ')
			if yourMoveStr in ['random', 'r']:
				return self.getRandomGoodAction(gameState)
			tokens = yourMoveStr.split(',')
			if len(tokens) != 2:
				continue
			for t in tokens:
				if not isinstance(t, int):
					continue
			action = tuple( [int(t) for t in tokens] )
			if action not in legalActions:
				print('Illegal action!')
				continue
			break
		print(action)
		return action

class AlphaBetaSearchAgent(Agent):

	def __init__(self, player):
		super(AlphaBetaSearchAgent, self).__init__(player)
		self.searchCount = 0

	def alphaBetaSearch(self, gameState, player, depth, alpha=-99999, beta=99999):
		#print(alpha, beta)
		self.searchCount += 1
		print(self.searchCount)
		if depth == 0 or gameState.isGoalState():
			return self.evaluationFunction(gameState, self.player, method='sspl'), []
			
		actions = gameState.getGoodActions()
		#print(alpha, beta)

		bestAction = None
		if player == self.player:
			bestScore = -999999
			for action in actions:
				nextState = gameState.getNextState(action)
				score, path = self.alphaBetaSearch(nextState, self.opponent, depth, alpha, beta)
				#print(player, action, score)
				if score > bestScore:
					bestScore = score
					bestAction = action
			
				if bestScore > beta:
					path.append(bestAction)
					return bestScore, path
				#print(alpha, beta, bestScore)
				alpha = max(alpha, bestScore)

		elif player == self.opponent:
			bestScore = 999999
			for action in actions:
				nextState = gameState.getNextState(action)
				score, path = self.alphaBetaSearch(nextState, self.player, depth-1, alpha, beta)
				#print(player, action, score)
				if score < bestScore:
					bestScore = score
					bestAction = action
				if bestScore < alpha:
					path.append(bestAction)
					return bestScore, path
				beta = min(beta, bestScore)
			
		path.append(bestAction)
		return bestScore, path.copy()	

	
	def getAction(self, gameState, maxDepth=2):
		player = self.player
		if gameState.getAlreadyPlayedActionsNum() < 4:
			maxDepth = 1
		
		self.searchCount = 0
		bestScore, bestPath = self.alphaBetaSearch(gameState, self.player, maxDepth)
		#print(self.searchCount)
		print(bestScore, bestPath)

		return bestPath[0]


class MyAgent1(Agent):

	def __init__(self, player):
		super(MyAgent1, self).__init__(player)
		self.randomAgent = RandomAgent(player)
		self.betterRandomAgent = BetterRandomAgent(player)
		self.minimaxSearchAgent = MinimaxSearchAgent(player)
		self.monteCarloSearchAgent = MonteCarloSearchAgent(player)

	def getAction(self, gameState):
		if gameState.getAlreadyPlayedActionsNum() < 10:
			return self.betterRandomAgent.getAction(gameState)
		else:
			return self.minimaxSearchAgent.getAction(gameState)

class MyAgent2(Agent):

	def __init__(self, player):
		super(MyAgent2, self).__init__(player)
		self.randomAgent = RandomAgent(player)
		self.betterRandomAgent = BetterRandomAgent(player)
		self.minimaxSearchAgent = AlphaBetaSearchAgent(player)
		self.monteCarloSearchAgent = MonteCarloSearchAgent(player)

	def getAction(self, gameState):
		if gameState.getAlreadyPlayedActionsNum() < 10:
			return self.betterRandomAgent.getAction(gameState)
		else:
			return self.monteCarloSearchAgent.getAction(gameState)



class MonteCarloNode(object):
	__slots__ = ('value', 'visits', 'heuristic')

	def __init__(self, h, value=0.0, visits=0):
		self.heuristic = h
		self.value = value
		self.visits = visits
		#self.parent = []
		#self.child = []

	def __str__(self):
		return('{ value: ' + str(self.value) + ', visits: ' + str(self.visits) + '}')

	def update(self, reward):
		self.visits += 1
		self.value += reward

	def getScore(self):
		return self.value / (self.visits or 1)

	def getProgressiveBias(self):
		return (self.heuristic+self.value) / (self.visits or 1)

'''
class MonteCarloTree():

	def __init__(self, root):
		self.root = None

	def addNode(self, node, parent):
		if self.root = None:
			self.root = node

		parent.child.append(node)
		node.parent.append(parent)

	def setRoot(self, node):
		self.root = node

	def deleteSubtree(self, node):
'''

def generateAMAFBackwardPropagation(explored, sizeThreshold):
	#print(explored)
	AMAF = set()
	for actionHistory in explored:
		blackHistory = actionHistory[0]
		blackNum = len(blackHistory)
		whiteHistory = actionHistory[1]
		whiteNum = len(whiteHistory)
		for i in range(blackNum+1):
			for j in range(whiteNum+1):
				if i + j < sizeThreshold:
					continue
				b = (itertools.combinations(blackHistory, i))
				w = (itertools.combinations(whiteHistory, j))
				for x in b:
					for y in w:
						AMAF.add( (frozenset(x),frozenset(y)) )

	return AMAF





def appendActionHistory(actionHistory, action, player):
	if player == HexPlayer.BLACK:
		return (actionHistory[0] | frozenset({action}), actionHistory[1]) 
	elif player == HexPlayer.WHITE:
		return (actionHistory[0], frozenset({action}) | actionHistory[1])


class MonteCarloSearchAgent(Agent):

	def __init__(self, player, **kwargs):
		super(MonteCarloSearchAgent, self).__init__(player)
		self.simulationTimeLimit = float(kwargs.get('time', 45))
		# self.simulationActionsLimit = int(kwargs.get('max_actions', 1000))
		# Exploration constant, increase for more exploratory actions,
		# decrease to prefer actions with known higher win rates.
		self.C = float(kwargs.get('C', 1.4))
		self.searchActions = set()
		self.tree = {} #defaultdict(MonteCarloNode)
		#self.tree = MonteCarloTree()

	def getAction(self, gameState):
		# Causes the AI to calculate the best action from the
		# current game state and return it.

		assert not gameState.isGoalState()

		self.maxDepth = 0
		#print(self.tree)

		for n in list(self.tree):
			if len(n[0]) + len(n[1]) <= gameState.getAlreadyPlayedActionsNum():
				del self.tree[n]

		#print(self.searchActions)

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
		actionHistory = gameState.actionHistory

		for a in gameState.getGoodActions():
			t = appendActionHistory(actionHistory, a, player)
			if t in self.tree:
				print(a, self.tree[t])
		#print(self.tree)
		bestAction = max( [ action for action in gameState.getGoodActions() if appendActionHistory(actionHistory, action, player) in self.tree ], 
				key=lambda x: self.tree[ appendActionHistory(actionHistory, x, player) ].getScore())

		print('Average reward:', self.tree[ appendActionHistory(actionHistory, bestAction, player) ].getScore())

		return bestAction

	def runSimulation(self, gameState):

		# Plays out a "random" game from the current position,
		# then updates the statistics tables with the result.

		# A bit of an optimization here, so we have a local
		# variable lookup instead of an attribute access each loop.
		
		explored = set()
		#print(gameState)
		currentState = gameState.copy()

		expand = True
		depth = 0
		firstPlayer = self.player
		secondPlayer = self.opponent
		player = firstPlayer

		# Action Path ( BLACK, WHITE )
		actionHistory = currentState.actionHistory
		minActionNum = currentState.getAlreadyPlayedActionsNum()

		
		#maxActionNum = len(legalActions)
		#print(maxActionNum)

		begin = time.time()
		while True:
			
			depth += 1
			#print(actionPath)
			#print(time.time()-begin)
			#print(legalActions)
			if depth == 1:
				actions = currentState.getGoodActions()
			else:
				actions = self.getReflexActions(currentState)

			if player == HexPlayer.BLACK:
				nextActionHistories = [ (actionHistory[0] | frozenset({action}), actionHistory[1]) for action in actions]
			elif player == HexPlayer.WHITE:
				nextActionHistories = [ (actionHistory[0], frozenset({action}) | actionHistory[1]) for action in actions]
			
			
			#print(time.time()-begin)
			#print(time.time()-begin)
			if all( nextActionHistory in self.tree for nextActionHistory in nextActionHistories):
				# UCB1
				logSum = log( sum( self.tree[ nextActionHistory ].visits for nextActionHistory in nextActionHistories ) or 1 )
				newActionHistory = max( [ nextActionHistory for nextActionHistory in nextActionHistories ], 
					key=lambda x: self.tree[x].getProgressiveBias() + self.C * sqrt(logSum / (self.tree[x].visits or 1)) )
			else:
				newActionHistory = random.choice(nextActionHistories)

			
			#print(newActionPath)
			
			if player == HexPlayer.BLACK:
				newAction = list(newActionHistory[0] - actionHistory[0])[0]
			elif player == HexPlayer.WHITE:
				newAction = list(newActionHistory[1] - actionHistory[1])[0]

			if expand and newActionHistory not in self.tree:
				expand = False
				self.tree[ newActionHistory ] = MonteCarloNode( self.evaluationFunction(currentState, firstPlayer) )
				self.maxDepth = max(depth, self.maxDepth)

			actionHistory = newActionHistory
			currentState.setToNextState(newAction, player)
			explored.add(newActionHistory)
			
			player = HexPlayer.OPPONENT(player)

			if currentState.isGoalState():
				break
		
		AMAF = generateAMAFBackwardPropagation(set(self.tree)&explored, minActionNum) & set(self.tree)
		
		#print(AMAF-explored)
		reward = currentState.getReward(firstPlayer)
		for path in AMAF:
			self.tree[path].update(reward)
		return reward



		




















