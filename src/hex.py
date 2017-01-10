import sys
import numpy as np
import time

class HexPlayer():
	BLACK = 1
	WHITE = 2

	EACH_PLAYER = [BLACK, WHITE]

class HexEnv():

	def __init__(self, boardSize, verbose=True):
		self.setHexStateParameter(boardSize)
		self.reset()
		self.verbose = verbose
		self.playerAgent = { 1: None, 2: None }

	def step(self, action, player):
		self.gameState = self.gameState.getNextState(action, player)

	def setHexStateParameter(self, boardSize):
		HexState.BOARD_SIZE = boardSize
		HexState.END_CELL = {
			1: (1, boardSize+1),
			2: (boardSize+1, 1)
		}

	def reset(self, boardSize=None):
		if boardSize != None:
			self.setHexStateParameter(boardSize)
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

	START_CELL = {
		1: (1, 0),
		2: (0, 1)
	}
	END_CELL = {
		1: None,
		2: None
	}

	def __init__(self):
		assert HexState.BOARD_SIZE != None
		self.board = {}
		for x in range(HexState.BOARD_SIZE):
			for y in range(HexState.BOARD_SIZE):
				self.board[(x+1,y+1)] = 0
		for i in range(HexState.BOARD_SIZE):
			self.board[(i+1, 0)] = 1
			self.board[(0, i+1)] = 2
			self.board[(i+1, HexState.BOARD_SIZE+1)] = 1
			self.board[(HexState.BOARD_SIZE+1, i+1)] = 2
		self.nextPlayer = 1
		self.winner = None
		self.lastAction = None
		self.dead = set()
		self.analysisResult = {
			'check': {
				1: [],
				2: []
			},
			'dead': {
				1: [],
				2: [],
				0: []
			},
			'captured': {
				1: [],
				2: []
			}
		}

	def copy(self):
		import copy
		state = copy.deepcopy(self)
		return state

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
		if len(goodActions) > 0:
			return goodActions

		return self.getLegalActions()

	def isLegalAction(self, action, player, prediction=False):
		if not prediction and player != self.nextPlayer:
			return False
		return action in self.getLegalActions()

	def getNextState(self, action, player=None, prediction=False):
		nextState = self.copy()
		if player == None:
			player = self.nextPlayer
		assert self.isLegalAction(action, player, prediction)
		nextState.board[action] = player
		nextState.nextPlayer = 3-player
		nextState.lastAction = action
		return nextState

	def getNeighbors(self, center, player=None, checkList=None):
		neighbors = []
		if checkList == None: checkList = self.getAllCells()
		for d in HexState.NEIGHBORNG_DIRECTION:
			c = (center[0]+d[0], center[1]+d[1])
			if c not in checkList:
				continue
			if player == None or self.board[c] == player:
				neighbors.append(c)
		return neighbors

	def getLegalNeighbors(self, center):
		return self.getNeighbors(center, 0)


	def getNeighborsPattern(self, center):
		board = self.board
		#allCells = self.getAllCells()

		neighborsPattern = []
		for dx, dy in HexState.NEIGHBORNG_DIRECTION:
			c = (center[0]+dx, center[1]+dy)
			if c not in self.board:
				neighborsPattern.append(0)
			else:	
				neighborsPattern.append(self.board[c])
		return neighborsPattern

	def isDead(self, coordinate):
		if  coordinate[0] == 0 or coordinate[1] == 0 \
			or coordinate[0] == HexState.BOARD_SIZE+1 or coordinate[1] == HexState.BOARD_SIZE+1:
			return False
		neighborsPattern = self.getNeighborsPattern(coordinate)
		if coordinate in self.dead:
			return True

		for pattern in self.DEAD_CELL_PATTERN:
			if isPatternsMatched(neighborsPattern, pattern):
				self.dead.add(coordinate)
				return True
		return False

	def getWinner(self):

		for player in [1,2]:
			if self.isWinner(player):
				return player
		legalActions = self.getLegalActions()
		if len(legalActions) == 0:
			self.winner = 0
			return 0
		return -1

	def getReward(self, player):
		winner = self.getWinner()
		opponent = 3-player
		if winner == player:
			return 1
		elif winner == opponent:
			return -1
		else:
			return 0

	def isWinner(self, player):
		if self.winner != None:
			return self.winner == player
		frontier = set()
		explored = set()
		frontier.add(HexState.START_CELL[player])

		while len(frontier) != 0:
			cell = frontier.pop()
			if cell == HexState.END_CELL[player]:
				self.winner = player
				return True
			for neighbor in self.getNeighbors(cell, player):
				if neighbor not in explored:
					frontier.add(neighbor)
			explored.add(cell)
		return False

	def isGoalState(self):
		if self.getWinner() != -1:
			return True
		return False

	def getMustPlayActions(self):
		legalActions = self.getLegalActions()
		player = self.nextPlayer
		opponent = 3-player
		
		# Winning Move
		winningActions = [action for action in legalActions 
			if self.getNextState(action, player).getWinner() == player]
		if len(winningActions) > 0:
			return winningActions
		
		# Not losing move
		notLosingActions = [action for action in legalActions
			if self.getNextState(action, opponent, True).getWinner() == opponent]
		if len(notLosingActions) > 0:
			return notLosingActions
		
		return []

	def isVulnerableToPlayer(self, center, player):
		if self.isDead(center):
			return False
		for neighbor in self.getLegalNeighbors(center):
			if self.getNextState(neighbor, player, True).isDead(center):
				return True
		return False

	def isCapturedByPlayer(self, center, player):
		for neighbor in self.getLegalNeighbors(center):
			if self.getNextState(neighbor, player, True).isDead(center) \
				and self.getNextState(center, player, True).isDead(neighbor):
				return True
		return False

	def analysis(self):
		begin = time.time()
		actionSet = set(self.getLegalActions())

		#print(time.time()-begin)
		blackWinningActions = set([action for action in actionSet
			if self.getNextState(action, HexPlayer.BLACK, True).getWinner() == HexPlayer.BLACK])
		actionSet -= blackWinningActions
		#print(time.time()-begin)
		whiteWinningActions = set([action for action in actionSet
			if self.getNextState(action, HexPlayer.WHITE, True).getWinner() == HexPlayer.WHITE])
		actionSet -= whiteWinningActions

		#print(time.time()-begin)
		blackDeadCells = set([cell for cell in self.getAllCells(HexPlayer.BLACK) if self.isDead(cell)])
		#print(time.time()-begin)
		whiteDeadCells = set([cell for cell in self.getAllCells(HexPlayer.WHITE) if self.isDead(cell)])
		#print(time.time()-begin)
		deadActions = set([action for action in actionSet if self.isDead(action)])
		actionSet -= deadActions
		#print(time.time()-begin)
		blackCapturedActions = set([action for action in actionSet if self.isCapturedByPlayer(action, HexPlayer.BLACK)])
		actionSet -= blackCapturedActions
		#print(time.time()-begin)
		whiteCapturedActions = set([action for action in actionSet if self.isCapturedByPlayer(action, HexPlayer.WHITE)])
		actionSet -= whiteCapturedActions

		#blackVulnerableActions = [action for action in actionSet if self.isVulnerableToPlayer(action, HexPlayer.BLACK)]
		#whiteVulnerableActions = [action for action in actionSet if self.isVulnerableToPlayer(action, HexPlayer.WHITE)]

		result = {
			'check': {
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
			}
		}
		return result
		'''
		print(result)
		print('Black Winning:', blackWinningActions)
		print('White Winning:', whiteWinningActions)
		print('Black Dead:', blackDeadCells)
		print('White Dead:', whiteDeadCells)
		print('Dead Actions:', deadActions)
		print('Black Captured: ', blackCapturedActions)
		print('White Captured: ', whiteCapturedActions)
		'''
		#print('Vulerable to Black: ', blackVulnerableActions)
		#print('Vulerable to White: ', whiteVulnerableActions)

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


