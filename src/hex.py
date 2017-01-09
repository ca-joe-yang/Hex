import sys
import numpy as np

'''
NOTHING = 0
BLACK = 1
WHITE = 2
'''

class HexEnv():

	def __init__(self, boardSize, verbose=True):
		self.reset(boardSize)
		self.verbose = verbose
		self.playerAgent = { 1: None, 2: None }

	def _step(self, action, player):
		assert self.toPlay == player, 'Wrong Player'
		self.gameState.makeMove(action, player)

	def reset(self, boardSize):
		self.gameState = HexGameState(boardSize)
		self.toPlay = 1

	def setPlayerAgent(self, player, agent):
		assert player in [1,2]
		self.playerAgent[player] = agent


	def autoPlay(self):
		while True:
			player = self.toPlay
			gameState = self.gameState
			#print(player)
			action = self.playerAgent[player](gameState)
			self._step(action, player)
			self.toPlay = 3-player

			if self.verbose:
				self.render()
			
			result = self.gameState.isGoalState()
			if result != -1:
				break

		return result

	def render(self):
		outfile = sys.stdout
		outfile.write(str(self.gameState))


class HexState:

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
		[2, 2, 2, 2, None, None],
		[1, 1, None, 2, 2, None],
		[1, None, 2, 2, 2, None],
		[2, None, 1, 1, 1, None],
	]

	START_CELL = {
		1: (1, 0)
		2: (0, 1)
	}
	END_CELL = {
		1: ()

	}

	def __init__(self, boardSize):
		self.N = boardSize
		self.board = {}
		for x in range(self.N):
			for y in range(self):
				self.board[(x+1,y+1)] == 0
		for i in range(self.N):
			self.board[(i+1, 0)] = 1
			self.board[(0, i+1)] = 2
			self.board[(i+1, self.N+1)] = 1
			self.board[(self.N+1, i+1)] = 2
		self.nextPlayer = 1

	def _copy(self):
		import copy
		state = copy.deepcopy(self)
		return state

	def __str__(self):

		board = self.board
		boardSize = self.N
		
		cellWidth = 5
		lineLength = cellWidth * (boardSize-1) + boardSize + 2

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

	def getAllCells(self, player=None):
		if not player:
			return [ c for c in self.board.keys() if self.board[c] == player]
		return self.board.keys()

	def getLegalActions(self):
		return self.getAllCells(0)

	def isLegalAction(self, action, player):
		if player != self.nextPlayer:
			return False
		return action in self.getLegalActions()

	def getNextState(self, action, player):
		nextState = self.copy()
		assert self.isLegalAction(action, player)
		nextState.board[action] = player
		return nextState

	def getNeighbors(self, center, player=None, checkList=None):
		neighbors = []
		if checkList == None: checkList = self.getAllCells()
		for d in self.NEIGHBORNG_DIRECTION:
			c = (center[0]+d[0], center[1]+d[1])
			if c not in checkList:
				continue
			if player == None or self.board[c] == player:
				neighbors.append(c)
		return neighbors

	def getLegalNeighbors(self, center):
		return getNeighbors(center, 0)


	def getNeighborsPattern(self, center):
		board = self.board
		#allCells = self.getAllCells()

		neighborsPattern = []
		for dx, dy in self.NEIGHBORNG_DIRECTION:
			c = (center[0]+dx, center[1]+dy)
			if (x,y) not in self.board:
				neighborsPattern.append(0)
			else:	
				neighborsPattern.append(self.board[c])
		return neighborsPattern

	def isDeadCell(self, coordinate):
		neighborsPattern = self.getNeighborsPattern(coordinate)

		for p in self.DEAD_CELL_PATTERN:
			pattern = p[:]
			for r in range(6):
				x = pattern.pop()
				pattern.insert(0, x)
				#print(coordinate, neighborsState, pattern)
				if isPatternsMatched(neighborsPattern, pattern):
					return True
		return False

	def getReward(self, player):
		
		frontier = set()
		explored = set()
		frontier.add(self.start[player])

		while len(frontier) != 0:
			node = frontier.pop()
			if node == self.end[player]:
				return True
			for neighbor in self.getNeighbors(node, player):
				if neighbor not in explored:
					frontier.add(neighbor)
			explored.add(node)
		return False


	def isGoalState(self):
		legalActions = self.getLegalActions()

		for player in [1,2]:
			if self.isPlayerWinner(player):
				return True
		if len(legalActions) == 0:
			return False

		return -1






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


		self.start = {
			1: (1, 0),
			2: (0, 1)
		}
		self.end = {
			1: (1, boardSize+1),
			2: (boardSize+1, 1)
		}

	def makeMove(self, action, player):
		if self.board[action[0], action[1]] == 0:
			self.board[action[0], action[1]] = player
			return True
		else: 
			return False

	def isGoalState(self):
		legalActions = self.getLegalActions()

		for player in [1,2]:
			if self.isPlayerWin(player):
				return player
		if len(legalActions) == 0:
			return 0

		return -1





	def isPlayerWin(self, player):






