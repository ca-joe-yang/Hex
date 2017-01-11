import sys
import numpy as np
import time
import itertools
import networkx as nx

class HexPlayer():
	BLACK = 1
	WHITE = 2

	EACH_PLAYER = [BLACK, WHITE]

class HexEnv():

	def __init__(self, boardSize, verbose=True):
		#self.setHexStateParameter(boardSize)
		self.reset(boardSize)
		self.verbose = verbose
		self.playerAgent = { 1: None, 2: None }

	def step(self, action, player):
		self.gameState = self.gameState.getNextState(action, player, False, False)

	def _setHexStateParameter(self, boardSize):
		N = boardSize
		HexState.BOARD_SIZE = N
		HexState.TARGET_CELL = {
			HexPlayer.BLACK: [(1, 0), (1, N+1)],
			HexPlayer.WHITE: [(0, 1), (N+1, 1)]
		}
		for i in range(N):
			HexState.VERTEX_MAPPING[(i+1, 0)] = 0
			HexState.VERTEX_MAPPING[(i+1, N+1)] = 1
			HexState.VERTEX_MAPPING[(0, i+1)] = 2
			HexState.VERTEX_MAPPING[(N+1, i+1)] = 3

		for x in range(N):
			for y in range(N):
				HexState.VERTEX_MAPPING[(x+1,y+1)] = 5*y + x + 4

	def reset(self, boardSize=None):
		if boardSize != None:
			self._setHexStateParameter(boardSize)
		#del self.gameState
		self.gameState = HexState()

	def setPlayerAgent(self, player, agent):
		assert player in [1,2]
		self.playerAgent[player] = agent


	def autoPlay(self):
		while not self.gameState.isGoalState():
			gameState = self.gameState
			player = gameState.nextPlayer
			agent = self.playerAgent[player]
			action = agent.getAction(gameState)
	
			self.step(action, player)

			if self.verbose:
				print(agent.getName(), 'move:', action)
				self.render()

	def getWinner(self):
		return self.gameState.getWinner()

	def render(self):
		outfile = sys.stdout
		outfile.write(str(self.gameState))


class HexState:

	BOARD_SIZE = None

	TARGET_CELL = {
		HexPlayer.BLACK: [],
		HexPlayer.WHITE: []
	}

	VERTEX_MAPPING = {}

	FIRST_PLAYER = 1  # Black
	SECOND_PLAYER = 2 # White

	NEIGHBORNG_DIRECTION = [ 
		(-1, 0), 
		(-1, 1),
		(0, 1),
		(1, 0),
		(1, -1),
		(0, -1)
	]

	DEAD_CELL_PATTERN = [ 
		[1, 1, 1, 1, None, None],
		[1, 1, 1, None, None, 1],
		[1, 1, None, None, 1, 1],
		[1, None, None, 1, 1, 1],
		[None, None, 1, 1, 1, 1],
		[None, 1, 1, 1, 1, None],
		
		[2, 2, 2, 2, None, None],
		[2, 2, 2, None, None, 2],
		[2, 2, None, None, 2, 2],
		[2, None, None, 2, 2, 2],
		[None, None, 2, 2, 2, 2],
		[None, 2, 2, 2, 2, None],
		
		[1, 1, None, 2, 2, None],
		[1, None, 2, 2, None, 1],
		[None, 2, 2, None, 1, 1],
		[2, 2, None, 1, 1, None],
		[2, None, 1, 1, None, 2],
		[None, 1, 1, None, 2, 2],
		
		[1, None, 2, 2, 2, None],
		[None, 2, 2, 2, None, 1],
		[2, 2, 2, None, 1, None],
		[2, 2, None, 1, None, 2],
		[2, None, 1, None, 2, 2],
		[None, 1, None, 2, 2, 2],

		[2, None, 1, 1, 1, None],
		[None, 1, 1, 1, None, 2],
		[1, 1, 1, None, 2, None],
		[1, 1, None, 2, None, 1],
		[1, None, 2, None, 1, 1],
		[None, 2, None, 1, 1, 1],
	]

	def __init__(self):
		assert HexState.BOARD_SIZE != None
		N = HexState.BOARD_SIZE
		self.board = {}

		for c in HexState.VERTEX_MAPPING:
			self.board[c] = 0

		for i in range(N):
			self.board[(i+1, 0)] = HexPlayer.BLACK
			self.board[(i+1, N+1)] = HexPlayer.BLACK
			self.board[(0, i+1)] = HexPlayer.WHITE
			self.board[(N+1, i+1)] = HexPlayer.WHITE
		
		self.nextPlayer = 1
		self.winner = None
		self.lastAction = None
		self.dead = set()
		self.captured = {
			HexPlayer.BLACK: set(),
			HexPlayer.WHITE: set()
		}
		self.shannonGraphs = {
			HexPlayer.BLACK: self._initShannonGraph(HexPlayer.BLACK),
			HexPlayer.WHITE: self._initShannonGraph(HexPlayer.WHITE)
		}

	def _initShannonGraph(self, player):
		N = HexState.BOARD_SIZE
		graph = nx.Graph()
		for x in range(N):
			for y in range(N):
				graph.add_node((x+1, y+1))
		for node in graph.nodes():
			for neighbor in HexState.getNeighbors(self.board, [node]):
				if neighbor in graph.nodes():
					graph.add_edge(node, neighbor)

		if player == HexPlayer.BLACK:
			node = (1, 0)
			graph.add_node(node)
			for i in range(N):
				neighbor = (i+1, 1)
				graph.add_edge(node, neighbor)
			node = (1, N+1)
			graph.add_node(node)
			for i in range(N):
				neighbor = (i+1, N)
				graph.add_edge(node, neighbor)
		elif player == HexPlayer.WHITE:
			node = (0, 1)
			graph.add_node(node)
			for i in range(N):
				neighbor = (1, i+1)
				graph.add_edge(node, neighbor)
			node = (N+1, 1)
			graph.add_node(node)
			for i in range(N):
				neighbor = (N, i+1)
				graph.add_edge(node, neighbor)
			
		return graph

	def _updateShannonGraphs(self, action, player):
		for p in HexPlayer.EACH_PLAYER:
			graph = self.shannonGraphs[p]
			#print(graph.nodes())
			neighbors = graph.neighbors(action)
			graph.remove_node(action)
			if player == p:
				for n1 in neighbors:
					for n2 in neighbors:
						if n1 != n2:
							graph.add_edge(n1, n2)
		#print(graph.number_of_nodes())
		#print(graph.number_of_edges())

	def copy(self):
		import copy
		state = copy.deepcopy(self)
		return state

	def _update(self, action, player, basic):

		self.lastAction = action
		self.board[action] = player
		self.nextPlayer = 3-player
		opponent = 3-player

		if len(HexState.getNeighbors(self.board, [action], player)) >= 2:
			explored = HexState.traverse(self.board, action, player)
			if all( target in explored for target in HexState.TARGET_CELL[player]):
				self.winner = player
		elif len(self.getLegalActions()) == 0:
			self.winner = 0

		self._updateShannonGraphs(action, player)

		if basic:
			return

		self._updateCaptured()
		self._updateDead(action)

	def _updateCaptured(self):
		for p in HexPlayer.EACH_PLAYER:
			zone = HexState.getNeighbors(self.board, self.getAllCells(p))
			#del self.captured
			self.captured[p] = set([ cell for cell in zone if HexState.checkIsCapturedByPlayer(self.board, cell, p)])

	def _updateDead(self, action):
		neighbors = HexState.getNeighbors(self.board, [action])
		for n in neighbors:
			if not self.isDead(n) and HexState.checkIsDead(self.board, n):
				self.dead.add(n)

	def __hash__(self):
		hashkey = []
		for x in range(HexState.BOARD_SIZE):
			for y in range(HexState.BOARD_SIZE):
				hashkey.append(self.board[(x+1,y+1)])
		return hash(tuple(hashkey))

	def __eq__(self, other):
		return self.__hash__() == other.__hash__()

	def __lt__(self, other):
		return self.__hash__() < other.__hash__()

	def __cmp__(self, other):
		if self.__hash__() < other.__hash__():
			return -1
		elif self.__hash__() > other.__hash__():
			return 1
		else:
			return 0

	def __str__(self):

		board = self.board
		boardSize = HexState.BOARD_SIZE
		
		cellWidth = 5
		lineLength = cellWidth * (boardSize+1) + boardSize + 2

		halfCellWidth = 3
		halfLineLength = int(lineLength / 2)
		halfBoardSize = int((boardSize-1) / 2)	

		lines = []

		indexStrList = [' ']
		for x in range(boardSize):
			indexStrList.append(str(x+1))
		lines.append('|  ' + '  |  '.join( indexStrList ) + '  |')	

		for y in range(boardSize):
			cellStrList = []
			for x in range(boardSize):
				c = (x+1, y+1)
				if board[c] == 0:
					cellStrList.append(' ')
				elif board[c] == 1:
					cellStrList.append('B')
				elif board[c] == 2:
					cellStrList.append('W')
			lines.append('|  ' + str(y+1)  + '  |  ' +  '  |  '.join( cellStrList ) + '  |')

		lines.insert(0, ' '*halfLineLength+'B')
		lines.append(' '*halfLineLength + 'B')

		offset = 0
		for i in range(len(lines)):
			if i > 1:
				offset += halfCellWidth
			if i == halfBoardSize+2:
				line = 'W'+' '*(cellWidth-1) + lines[i] + ' '*(cellWidth-1)+'W'
			else:
				line = ' '*cellWidth + lines[i]
			line =  ' '*offset + line
			if i != len(lines) - 1:
				line += '\n' + ' '*offset + '-'*lineLength
			lines[i] = line[5:]

		lines.append('')

		return ('\n'.join(lines))

	def getAlreadyPlayedActionsNum(self):
		return HexState.BOARD_SIZE ** 2 - len(self.getLegalActions())

	def getAllCells(self, player=None):
		if player != None:
			return [ c for c in self.board.keys() if self.board[c] == player]
		return self.board.keys()

	def getLegalActions(self):
		return self.getAllCells(0)

	def getGoodActions(self):
		goodActions = [action for action in self.getLegalActions()
			if not self.isDead(action)]
		#print(goodActions)
		if len(goodActions) > 0:
			return goodActions

		return self.getLegalActions()

	def isLegalAction(self, action, player, prediction=False):
		if not prediction and player != self.nextPlayer:
			return False
		return action in self.getLegalActions()

	def getNextState(self, action, player=None, prediction=False, basic=True):
		nextState = self.copy()
		if player == None:
			player = self.nextPlayer
		assert self.isLegalAction(action, player, prediction)
		nextState._update(action, player, basic)
		return nextState

	def isDead(self, cell):
		return cell in self.dead

	def isCapturedByPlayer(self, cell, player):
		return cell in self.captured[player]

	def isGoalState(self):
		return self.winner != None

	def isWinner(self, player):
		return self.winner == player

	def getWinner(self):
		return self.winner

	def getReward(self, player):
		opponent = 3-player
		if self.winner == player:
			return 1
		elif self.winner == opponent:
			return -1
		else:
			return 0

	'''
	def isVulnerableToPlayer(self, center, player):
		if self.isDead(center):
			return False
		for neighbor in self.getLegalNeighbors(center):
			if self.getNextState(neighbor, player, True).isDead(center):
				return True
		return False
	'''







	def analysis(self):
		N = HexState.BOARD_SIZE
		begin = time.time()
		actionSet = set(self.getGoodActions())

		blackWinningActions = HexState.getWinningActions(self.board, HexPlayer.BLACK)
		whiteWinningActions = HexState.getWinningActions(self.board, HexPlayer.WHITE)

		actionSet -= whiteWinningActions
		actionSet -= blackWinningActions

		blackCells = self.getAllCells(HexPlayer.BLACK)
		whiteCells = self.getAllCells(HexPlayer.WHITE)
		blackDeadCells = set( [cell for cell in self.dead if cell in blackCells] )
		whiteDeadCells = set( [cell for cell in self.dead if cell in whiteCells] )
		deadActions = set( [action for action in self.dead if action in actionSet] )
		actionSet -= deadActions
		#print(time.time()-begin)
		blackCapturedActions = set([action for action in HexState.getNeighbors(self.board, blackCells) if self.isCapturedByPlayer(action, HexPlayer.BLACK)])
		actionSet -= blackCapturedActions
		#print(time.time()-begin)
		whiteCapturedActions = set([action for action in HexState.getNeighbors(self.board, whiteCells) if self.isCapturedByPlayer(action, HexPlayer.WHITE)])
		actionSet -= whiteCapturedActions
		#print(time.time()-begin)

		#blackVulnerableActions = [action for action in actionSet if self.isVulnerableToPlayer(action, HexPlayer.BLACK)]
		#whiteVulnerableActions = [action for action in actionSet if self.isVulnerableToPlayer(action, HexPlayer.WHITE)]

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
		return result
		#print('Vulerable to Black: ', blackVulnerableActions)
		#print('Vulerable to White: ', whiteVulnerableActions)
	

	@staticmethod
	def checkIsCapturedByPlayer(board, center, player):
		for neighbor in HexState.getNeighbors(board, [center], 0):
			board1 = board.copy()
			board1[neighbor] = player
			board2 = board.copy()
			board2[center] = player
			if HexState.checkIsDead(board1, center) and HexState.checkIsDead(board2, neighbor):
				return True
		return False
	
	@staticmethod
	def checkIsDead(board, center):
		N = HexState.BOARD_SIZE
		if center[0] == 0 or center[1] == 0 or center[0] == N+1 or center[1] == N+1:
			return False

		neighborsPattern = []
		for dx, dy in HexState.NEIGHBORNG_DIRECTION:
			c = (center[0]+dx, center[1]+dy)
			if c not in board:
				neighborsPattern.append(0)
			else:	
				neighborsPattern.append(board[c])

		for pattern in HexState.DEAD_CELL_PATTERN:
			if isPatternsMatched(neighborsPattern, pattern):
				return True
		return False

	@staticmethod
	def getNeighbors(board, actions, player=None, radius=1):
		neighbors = set(actions)
		for r in range(radius):
			tmp = neighbors.copy()
			for action in neighbors:
				for d in HexState.NEIGHBORNG_DIRECTION:
					c = (action[0]+d[0], action[1]+d[1])
					if c not in board:
						continue
					if player == None or board[c] == player:
						tmp.add(c)
			neighbors |= tmp
		for a in actions:
			neighbors.remove(a)
		return neighbors

	@staticmethod
	def traverse(board, action, player):
		frontier = set()
		explored = set()
		frontier.add(action)

		while len(frontier) != 0:
			cell = frontier.pop()
			for neighbor in HexState.getNeighbors(board, [cell], player):
				if neighbor not in explored:
					frontier.add(neighbor)
			explored.add(cell)
		return explored

	@staticmethod
	def getInfluenceZone(board, action, player):
		explored = HexState.traverse(board, action, player)
		return HexState.getNeighbors(board, explored, 0)
	
	@staticmethod
	def getWinningActions(board, player):
		zone1 = HexState.getInfluenceZone(board, HexState.TARGET_CELL[player][0], player)
		zone2 = HexState.getInfluenceZone(board, HexState.TARGET_CELL[player][1], player)

		return zone1 & zone2

	@staticmethod
	def getPlayerBridgesNum(board, player):
		count = 0
		for cell in board:
			if board[cell] != player:
				continue
			cx, cy = cell
			for i in range(6):
				j = (i+1) % 6
				dx1, dy1 = HexState.NEIGHBORNG_DIRECTION[i]
				dx2, dy2 = HexState.NEIGHBORNG_DIRECTION[j]
				river1 = (cx + dx1, cy + dy1)
				river2 = (cx + dx2, cy + dy2)
				bridge = (cx + dx1 + dx2, cy + dy1 + dy2)
				if bridge not in board or board[bridge] != player:
					continue
				if river1 in board and board[river1] == 0 and river2 in board and board[river2] == 0:
					count += 1
		return int(count / 2)



def isNeighbor(coordinate1, coordinate2):
	x1, y1 = coordinate1
	x2, y2 = coordinate2

	dx = x1 - x2
	dy = y1 - y2

	return direction in HexState.NEIGHBORNG_DIRECTION

def isPatternsMatched(pattern, targetPattern):
	assert len(pattern) == len(targetPattern)
	for p1, p2 in zip(pattern, targetPattern):
		if p2 == None:
			continue
		if p1 != p2:
			return False
	return True


