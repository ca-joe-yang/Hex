import sys
import numpy as np
import time
import itertools
import networkx as nx
import copy

class HexPlayer():
	BLACK = 1
	WHITE = 2

	EACH_PLAYER = [BLACK, WHITE]

	@staticmethod
	def OPPONENT(player):
		if player == HexPlayer.BLACK:
			return HexPlayer.WHITE
		elif player == HexPlayer.WHITE:
			return HexPlayer.BLACK
		return None

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
			begin = time.time()
			action = agent.getAction(gameState)
	
			self.step(action, player)

			if self.verbose:
				print('Response Time: ', time.time()-begin)
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
		self.N = N
		self.board = {}
		self.legalActions = []
		self.goodActions = []
		self.actionHistory = {
			HexPlayer.BLACK: frozenset(),
			HexPlayer.WHITE: frozenset()
		}

		for x in range(N):
			for y in range(N):
				c = (x+1, y+1)
				self.board[c] = 0
				self.legalActions.append(c)
				self.goodActions.append(c) 


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
		#self.goodMoves = []

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
		begin = time.time()
		state = copy.deepcopy(self)
		#print(time.time()-begin)
		return state

	def _update(self, action, player, basic):

		begin = time.time()
		self.lastAction = action
		self.legalActions.remove(action)
		self.goodActions.remove(action)
		self.actionHistory[player] |= frozenset({player})
		self.board[action] = player
		self.nextPlayer = 3-player
		opponent = 3-player

		self._updateShannonGraphs(action, player)
		# Check Goal State
		try:
			shortest = nx.shortest_path_length(
				self.shannonGraphs[player], 
				source=HexState.TARGET_CELL[player][0], 
				target=HexState.TARGET_CELL[player][1],
			)
			if shortest == 1:
				self.winner = player
		except:
			pass
		#print('update',time.time()-begin)

		if basic:
			return


		self._updateDead(action)
		#self._updateCaptured()
		#print(time.time()-begin)

	def _updateCaptured(self):
		for p in HexPlayer.EACH_PLAYER:
			zone = HexState.getNeighbors(self.board, self.getAllCells(p))
			for cell in zone:
				if cell in self.captured[p]:
					if cell in self.dead:
						self.captured[p].remove(cell)
				else:
					if HexState.checkIsCapturedByPlayer(self.board, cell, p):
						self.captured[p].add(cell)
			#del self.captured
			#self.captured[p] = set([ cell for cell in zone if HexState.checkIsCapturedByPlayer(self.board, cell, p)])

	def _updateDead(self, action):
		neighbors = HexState.getNeighbors(self.board, [action])
		for n in neighbors:
			if not self.isDead(n) and HexState.checkIsDead(self.board, n):
				
				self.dead.add(n)
				if n in self.goodActions:
					self.goodActions.remove(n)

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
				if board[c] == HexPlayer.BLACK:
					cellStrList.append('B')
				elif board[c] == HexPlayer.WHITE:
					cellStrList.append('W')
				elif self.isDead(c):
					cellStrList.append('X')
				elif board[c] == 0:
					cellStrList.append(' ')
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
		return self.legalActions

	def getGoodActions(self):
		return self.goodActions



	def isLegalAction(self, action, player, prediction=False):
		if not prediction and player != self.nextPlayer:
			return False
		return action in self.getLegalActions()

	def getCells(self, **kwargs):
		candidates = self.board.keys()

		player = kwargs.pop('player')
		assert player in ['all', 'legal', HexPlayer.BLACK, HexPlayer.WHITE]

		if player == 'legal':
			candidates = [ c for c in candidates if self.board[c] == 0 ]
		elif not player == 'all':
			candidates = [ c for c in candidates if self.board[c] == player ]

		dead = kwargs.pop('dead', None)
		assert dead in [True, False, None]
		if dead != None:
			candidates = [ c for c in candidates if self.isDead(c) == dead ]

		captured = kwargs.pop('captured', None)
		assert captured in [True, False, None]
		if captured != None:
			candidates_tmp = [ c for c in candidates if self.isCaptured(c) == captured ]
			if len(candidates_tmp) != 0:
				candidates = candidates_tmp

		assert len(kwargs) == 0
		#vulnerable = bool(kwargs.get('dead', False))
			

		return candidates

	def getNextState(self, action, player=None, prediction=False, basic=True):
		begin=time.time()
		nextState = self.copy()
		#print('copy',time.time()-begin)
		if player == None:
			player = self.nextPlayer
		assert self.isLegalAction(action, player, prediction)

		nextState._update(action, player, basic)
		#print(time.time()-begin)
		return nextState

	def setToNextState(self, action, player):
		self._update(action, player, False)

	def getEndStateReward(self, actionPaths, player):
		playerActions = actionPaths[0]
		opponentActions = actionPaths[1]

		endState = self.copy()
		
		p = endState.nextPlayer
		for action in playerActions:
			endState.board[action] = player
			endState._updateShannonGraphs(action, player)

		opponent = HexPlayer.OPPONENT(player)
		for action in opponentActions:
			endState.board[action] = opponent
			endState._updateShannonGraphs(action, opponent)

		for p in HexPlayer.EACH_PLAYER:
			try:
				shortest = nx.shortest_path_length(
					endState.shannonGraphs[p], 
					source=HexState.TARGET_CELL[p][0], 
					target=HexState.TARGET_CELL[p][1],
				)
				if shortest == 1:
					winner = p
					break
			except:
				pass
		#print(winner)
		#print(endState, winner)
		if winner == player:
			return 1
		elif winner == HexPlayer.OPPONENT(player):
			return -1



	def isDead(self, cell):
		return cell in self.dead

	def isCapturedByPlayer(self, cell, player):
		return cell in self.captured[player]

	def isCaptured(self, cell):
		for p in HexPlayer.EACH_PLAYER:
			if self.isCapturedByPlayer(cell, p):
				return True
		return False

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

	def getShortestDistanceSum(self, player):
		graph = self.shannonGraphs[player]

		target = HexState.TARGET_CELL[player][0]
		distances1 = nx.single_source_shortest_path_length(graph, source=target)
		del distances1[target]
		
		target = HexState.TARGET_CELL[player][1]
		distances2 = nx.single_source_shortest_path_length(graph, source=target)
		del distances2[target]

		shortestDistanceSum = {}
		for cell in self.getCells(player='legal'):
			try:
				shortestDistanceSum[cell] = distances1[cell] + distances2[cell]
			except:
				pass
		return shortestDistanceSum
		

	'''
	def isVulnerableToPlayer(self, center, player):
		if self.isDead(center):
			return False
		for neighbor in self.getLegalNeighbors(center):
			if self.getNextState(neighbor, player, True).isDead(center):
				return True
		return False
	'''







	
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

	@staticmethod
	def getHashKeyFromBoard(board):
		hashkey = []
		for x in range(HexState.BOARD_SIZE):
			for y in range(HexState.BOARD_SIZE):
				hashkey.append(board[(x+1,y+1)])
		return hash(tuple(hashkey))

	'''
	@staticmethod
	def getNextBoard(board, action, player):
		b = []
		for x in range(HexState.BOARD_SIZE):
			for y in range(HexState.BOARD_SIZE):
				c = (x+1, y+1)
				if c == action:
					b.append(player)
				else:
					b.append(board[c])
		#b[action] = player
		return b
	'''

	@staticmethod
	def getNextBoardHashKey(board, action, player):
		hashkey = []
		begin=time.time()
		for x in range(HexState.BOARD_SIZE):
			for y in range(HexState.BOARD_SIZE):
				if (x+1, y+1) == action:
					hashkey.append(player)
				else:
					hashkey.append(board[(x+1,y+1)])
		#print(time.time()-begin)
		return hash(tuple(hashkey))

		#return HexState.getHashKeyFromBoard(HexState.getNextBoard(board, action, player))





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


