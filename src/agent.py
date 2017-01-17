import time
import random
import numpy as np
from abc import ABCMeta, abstractmethod
from math import log, sqrt
from collections import defaultdict
import networkx as nx
from hex import HexPlayer, HexState
import itertools
import _pickle as pickle



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

    def getAttackActions(self, gameState, player):
        graph = gameState.shannonGraphs[player]
        paths = nx.all_shortest_paths(graph, HexState.TARGET_CELL[player][0], HexState.TARGET_CELL[player][1])

        actions = set()
        for p in paths:
            actions |= set(p)
        goodActions = gameState.getGoodActions()
        actions &= set(goodActions)
        actions = list(actions)

        notNeighborActions = []
        for a in actions:
            if not a in HexState.getNeighbors(gameState.board, list(gameState.actionHistory[player-1])):
                if not a in HexState.getNeighbors(gameState.board, gameState.getBorder()):
                    notNeighborActions.append(a)
        if len(notNeighborActions) != 0:
            return notNeighborActions
        else:
            return actions

    def getDefenseActions(self, gameState, player):
        graph = gameState.shannonGraphs[player]
        paths = nx.all_shortest_paths(graph, HexState.TARGET_CELL[player][0], HexState.TARGET_CELL[player][1])

        actions = set()
        for p in paths:
            actions |= set(p)
        goodActions = gameState.getGoodActions()
        actions &= set(goodActions)

        return list(actions)

    def getMustWinActions(self, player):
        graph = gameState.shannonGraphs[player]
        paths = nx.all_shortest_paths(graph, HexState.TARGET_CELL[player][0], HexState.TARGET_CELL[player][1])
        if len(paths[0]) > 2:
            return None
        actions = []
        for p in paths:
            actions.append(p[0])
        return actions

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

class OnlyAttackAgent(Agent):

    def __init__(self, player):
        super(OnlyAttackAgent, self).__init__(player)

    def getAction(self, gameState):
        return random.choice(self.getAttackActions(gameState, self.player))

class OnlyDefenseAgent(Agent):

    def __init__(self, player):
        super(OnlyDefenseAgent, self).__init__(player)

    def getAction(self, gameState):
        return random.choice(self.getDefenseActions(gameState, self.player))

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

        return bestPath[0]

class MonteCarloNode(object):
    #__slots__ = ('value', 'visits', 'heuristic', 'children')

    def __init__(self, h, value=0.0, visits=0):
        self.heuristic = h
        self.value = value
        self.visits = visits
        #self.children = []

    def __str__(self):
        return('{ value: ' + str(self.value) + ', visits: ' + str(self.visits) + ', average: ' + str(self.getAverageValue()) + '}')

    def __repr__(self):
        return('{ value: ' + str(self.value) + ', visits: ' + str(self.visits) + ', average: ' + str(self.getAverageValue()) + '}')

    def update(self, reward):
        self.visits += 1
        self.value += reward

    def getAverageValue(self):
        return self.value / (self.visits or 1)

    def getProgressiveBias(self):
        return self.getAverageValue()
        if self.visits >= 10:
            return (self.heuristic+self.value) / (self.visits or 1)
        else:
            return self.getAverageValue()

    '''
    def addChildNode(self, node):
    try:
    self.children
    except:
    self.children = []
    self.children.append(node)

    def getBestChild(self):
    pass
    '''
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

def appendActionHistory(actionHistory, action, player):
    if player == HexPlayer.BLACK:
        return (actionHistory[0] | frozenset({action}), actionHistory[1]) 
    elif player == HexPlayer.WHITE:
        return (actionHistory[0], frozenset({action}) | actionHistory[1])


class MonteCarloSearchAgent(Agent):

    def __init__(self, player, **kwargs):
        super(MonteCarloSearchAgent, self).__init__(player)
        self.simulationTimeLimit = float(kwargs.get('time', 60))
        # self.simulationActionsLimit = int(kwargs.get('max_actions', 1000))
        # Exploration constant, increase for more exploratory actions,
        # decrease to prefer actions with known higher win rates.
        self.C = float(kwargs.get('C', 1.4))
        self.dataFilename = kwargs.pop('filename')
        self.mode = kwargs.pop('mode')
        self.tree = self._loadTree()

    def __del__(self):
        #pass
        if self.mode == 'train':
            self._saveTree()

    def _saveTree(self):
        with open(self.dataFilename, 'wb') as f:
            #print (self.tree)
            #print (f)
            pickle.dump(self.tree, f, protocol=4)

    def _loadTree(self):
        try:
            with open(self.dataFilename, 'rb') as f:
                return pickle.load(f)
        except:
            return {}

    def getAction(self, gameState):
        # Causes the AI to calculate the best action from the
        # current game state and return it.

        assert not gameState.isGoalState()

        self.maxDepth = 0

        simulationCount = 0
        if self.mode == 'train':
            beginTime = time.time()
            while time.time() - beginTime < self.simulationTimeLimit:
                simulationCount += 1
                reward = self.runSimulation(gameState)
                print('Simulation', simulationCount, ', reward:', reward, '  ', end='\r')

            print()
            print('Simulation counts:', simulationCount)
            print('Search max depth:', self.maxDepth)
            print('Time elapsed:', time.time() - beginTime)

        player = gameState.nextPlayer
        actionHistory = gameState.actionHistory
        actions = self.getReflexActions(gameState)

        for action in actions:
            t = HexState.generateNextBoard(HexState.convertBoard2BoardStr(gameState.board), action, player)
            if t in self.tree:
                print(action, self.tree[t])

        bestAction = max( [ action for action in actions if HexState.convertBoard2BoardStr(gameState.getNextState(action, player).board) in self.tree ], 
                         key=lambda x: self.tree[ HexState.convertBoard2BoardStr(gameState.getNextState(x, player).board) ].getAverageValue())

        print('Average reward:', self.tree[ HexState.convertBoard2BoardStr(gameState.getNextState(bestAction, player).board) ].getAverageValue())
        #self._saveTree()
        return bestAction

    def runSimulation(self, gameState):

        # Plays out a "random" game from the current position,
        # then updates the statistics tables with the result.

        # A bit of an optimization here, so we have a local
        # variable lookup instead of an attribute access each loop.

        lastNodeKey = None

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

        initBoard = HexState.convertBoard2BoardStr(currentState.board)
        board = initBoard
        playOutPath = []
        actionPath = []

        begin = time.time()
        while True:

            depth += 1
            if depth == 1:
                actions = currentState.getGoodActions()
            else:
                actions = self.getReflexActions(currentState)
                if len(actions) == 0:
                    return list(set(self.getAttackActions(currentState, player) + self.getDefenseActions(currentState, player)))

            #if board in self.tree:
            playOutPath.append((board, player))

            player = currentState.nextPlayer
            nextBoards = [ HexState.generateNextBoard(board, action, player) for action in actions ]

            '''
            if player == HexPlayer.BLACK:
            nextActionHistories = [ (actionHistory[0] | frozenset({action}), actionHistory[1]) for action in actions]
            elif player == HexPlayer.WHITE:
            nextActionHistories = [ (actionHistory[0], frozenset({action}) | actionHistory[1]) for action in actions]
            '''
            if all( nextBoard in self.tree for nextBoard in nextBoards):
                # Upper Confidence Bound
                visitSum = sum( self.tree[nextBoard].visits for nextBoard in nextBoards )
                #visitSum = self.tree[ board ].visits
                logSum = log( visitSum or 1 )
                newBoard = max( nextBoards, 
                               key=lambda x: self.tree[x].getProgressiveBias() + self.C * sqrt(logSum / (self.tree[x].visits or 1)) )
                newAction, checkPlayer = HexState.getNewActionFromBoards(board, newBoard)
                assert checkPlayer == player
            else:
                newAction = random.choice(actions)
                newBoard = HexState.generateNextBoard(board, newAction, player)

            #newAction, checkPlayer = HexState.getNewActionFromBoards(board, newBoard)
            #assert checkPlayer == player
            #print(newAction)

            #print(newActionPath)

            #actionHistory = newActionHistory
            currentState.setToNextState(newAction, player)

            if expand and newBoard not in self.tree:
                expand = False
                self.tree[ newBoard ] = MonteCarloNode( self.evaluationFunction(currentState, firstPlayer) )
                #self.tree[ board ].addChildNode( self.tree[newBoard] )
                self.maxDepth = max(depth, self.maxDepth)

            board = newBoard
            actionPath.append(newAction)

            player = HexPlayer.OPPONENT(player)

            if currentState.isGoalState():
                playOutPath.append((board, player))
                break

        endBoard = board
        reward = currentState.getReward(firstPlayer)

        #print(playOutPath)
        #print(actionPath)

        # AMAF
        if HexState.BOARD_SIZE > 6:
            for i in range(len(playOutPath)):
                b, p = playOutPath[i]
                if i < HexState.BOARD_SIZE-6:
                    for a in actionPath:
                        if a == actionPath[0]:
                            continue
                        sibling = HexState.generateNextBoard(b, a, p)
                        if sibling in self.tree:
                            self.tree[sibling].update(reward)
                    if len(actionPath) > 0:
                        actionPath.remove(actionPath[0])

        for b, p in playOutPath:
            if b in self.tree:
                self.tree[b].update(reward)

        return reward

