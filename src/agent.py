import time
import random
import numpy as np
from abc import ABCMeta, abstractmethod
from math import log, sqrt
from collections import defaultdict
import networkx as nx
from hex import HexState 

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
		self.searchActions = set()
		self.tree = {} #defaultdict(MonteCarloNode)

	def getAction(self, gameState):
		# Causes the AI to calculate the best action from the
		# current game state and return it.

		assert not gameState.isGoalState()

		self.maxDepth = 0
		legalActions = gameState.getLegalActions()
		for a in self.searchActions - set(legalActions):
			for n in list(self.tree):
				if a in n[0] or a in n[1]:
					del self.tree[n]
		self.searchActions = set(legalActions.copy())
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

		for a in gameState.getGoodActions():
			t = (frozenset({a}), frozenset())
			if t in self.tree:
				print(a, self.tree[t])
		#print(self.tree)
		bestAction = max( [ action for action in gameState.getGoodActions() if (frozenset({action}), frozenset()) in self.tree ], 
				key=lambda x: self.tree[ (frozenset({x}), frozenset()) ].getScore())

		print('Average reward:', self.tree[ (frozenset({bestAction}), frozenset()) ].getScore())

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
		actionPath = (gameState.actionHistory[firstPlayer], gameState.actionHistory[secondPlayer])

		board = gameState.board
		#maxActionNum = len(legalActions)
		#print(maxActionNum)

		begin = time.time()
		while True:
			begin = time.time()
			depth += 1
			#print(actionPath)
			#print(time.time()-begin)
			#print(legalActions)
			'''
			if player == firstPlayer:
				nextActionPaths = [ (actionPath[0] | frozenset({action}), actionPath[1]) for action in currentState.getGoodActions()]
			elif player == secondPlayer:
				nextActionPaths = [ (actionPath[0], frozenset({action}) | actionPath[1]) for action in currentState.getGoodActions()]
			'''
			print('a',time.time()-begin)
			goodActions = currentState.getGoodActions()
			print(time.time()-begin)
			nextBoards = [ HexState.getNextBoardHashKey(board, action, player) for action in goodActions]
			print('x',time.time()-begin)
			#print(time.time()-begin)
			#print(time.time()-begin)
			if all( nextBoard in self.tree for nextBoard in nextBoards):
				# UCB1
				logSum = log( 
					sum( 
						self.tree[ nextBoard ].visits \
						for nextBoard in nextBoards 
					) or 1 
				)
				newActionIndex = argmax( 
					[ nextBoard for nextBoard in nextBoards ], 
					key=lambda x: self.tree[ x ].getScore() + \
					self.C * sqrt(logSum / (self.tree[x].visits or 1)) 
				)
				newAction = goodActions[newActionIndex]
			else:
				newAction = random.choice(goodActions)
			print(time.time()-begin)
			
			#print(newAction)
			
			'''
			if player == firstPlayer:
				newAction = list(newActionPath[0] - actionPath[0])[0]
				#player = secondPlayer
			elif player == secondPlayer:
				newAction = list(newActionPath[1] - actionPath[1])[0]
				#player = firstPlayer
			'''

			#actionPath = newActionPath
			currentState.setToNextState(newAction, player)
			print('e',time.time()-begin)
			if expand and currentState not in self.tree:
				expand = False
				self.tree[ currentState ] = MonteCarloNode()
				self.maxDepth = max(depth, self.maxDepth)
			explored.add(currentState)
			print(time.time()-begin)
			
			if player == firstPlayer:
				player = secondPlayer
			elif player == secondPlayer:
				player = firstPlayer

			if currentState.isGoalState():
				break
		print(time.time()-begin)
		reward = currentState.getReward(firstPlayer)
		for path in explored:
			if path in self.tree:
				self.tree[path].update(reward)
		print(time.time()-begin)
		return reward



		




















