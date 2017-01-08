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

	def reset(self, boardSize):
		self.gameState = HexGameState(boardSize)
		self.toPlay = 1

	def setPlayerAgent(self, player, agent):
		assert player in [1,2]
		self.playerAgent[player] = agent

	def _step(self, action, player):
		assert self.toPlay == player, 'Wrong Player'
		self.gameState.makeMove(action, player)

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


class HexGameState:

	def __init__(self, boardSize):
		self.board = np.zeros((boardSize+2, boardSize+2), dtype=np.int32)
		self.board[:, 0] = 1
		self.board[0, :] = 2
		self.board[boardSize+1, :] = 2
		self.board[:, boardSize+1] = 1

		self.neighboringDirection = [ 
			(-1, 0), 
			(-1, 1),
			(0, 1),
			(1, 0),
			(1, -1),
			(0, -1)
		]

		self.start = {
			1: (1, 0),
			2: (0, 1)
		}
		self.end = {
			1: (1, boardSize+1),
			2: (boardSize+1, 1)
		}

	def	_getAllCells(self, player=None):
		cellsList = []
		for x in range(self.board.shape[0]):
			for y in range(self.board.shape[1]):
				if player == None or self.board[x, y] == player:
					cellsList.append((x,y))
		return cellsList

	def makeMove(self, action, player):
		if self.board[action[0], action[1]] == 0:
			self.board[action[0], action[1]] = player
			return True
		else: 
			return False

	def getSuccessorStates(self, player):
		return

	def isGoalState(self):
		legalActions = self.getLegalActions()

		for player in [1,2]:
			if self.isPlayerWin(player):
				return player
		if len(legalActions) == 0:
			return 0

		return -1


	def getAllLegalCells(self):
		cellsList = []
		for x in range(1, self.board.shape[0]-1):
			for y in range(1, self.board.shape[1]-1):
				cellsList.append((x,y))
		return cellsList

	def getLegalActions(self):
		return self._getAllCells(0)

	def getNeighbors(self, center, player=None, onlyList=None):
		neighbors = []
		if onlyList == None: onlyList = self._getAllCells()
		for n in onlyList:
			if not self.isNeighbor(center, n):
				continue
			if player == None or self.board[n[0], n[1]] == player:
				neighbors.append(n)
		return neighbors

	def getLegalNeighbors(self, center):
		return getNeighbors(center, 0, self.getAllLegalCells())

	def isNeighbor(self, coordinate1, coordinate2):
		x1, y1 = coordinate1
		x2, y2 = coordinate2

		dx = x1 - x2
		dy = y1 - y2
		for c in self.neighboringDirection:
			if (dx, dy) == c:
				return True
		return False

	def isPlayerWin(self, player):
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

	def isDeadCell(self, coordinate):
		deadCellPattern = [ 
			[1, 1, 1, 1, None, None],
			[2, 2, 2, 2, None, None],
			[1, 1, None, 2, 2, None],
			[1, None, 2, 2, 2, None],
			[2, None, 1, 1, 1, None],
		]

		neighborsState = self.getNeighborsState(coordinate)

		for p in deadCellPattern:
			pattern = p[:]
			for r in range(6):
				x = pattern.pop()
				pattern.insert(0, x)
				print(coordinate, neighborsState, pattern)
				if self.isNeighborsStateMatchPattern(neighborsState, pattern):
					return True
		return False

	def getNeighborsState(self, center):
		board = self.board
		allCells = self.getAllCells()

		neighborsState = []
		for dx, dy in self.neighboringDirection:
			x = center[0] + dx
			y = center[1] + dy
			if (x,y) not in allCells:
				neighborsState.append(0)
			else:	
				neighborsState.append(board[x][y])
		return neighborsState

	def isNeighborsStateMatchPattern(self, neighborsState, pattern):

		for i, j in zip(neighborsState, pattern):
			if j == None:
				continue
			if i != j:
				return False
		return True


	def __str__(self):
		board = self.board
		boardSize = self.board.shape[0]
		cellWidth = 5

		lineLength = cellWidth * (boardSize-1) + boardSize + 2

		halfCellWidth = 3
		halfLineLength = int(lineLength / 2)
		halfBoardSize = int((boardSize-1) / 2)	

		lines = []

		indexStrList = [' ']
		for x in range(boardSize-2):
			indexStrList.append(str(x+1))
		lines.append('|  ' + '  |  '.join( indexStrList ) + '  |')	

		for y in range(1,boardSize-1):
			cellStrList = []
			for x in range(1,boardSize-1):
				if board[x, y] == 0:
					cellStrList.append(' ')
				elif board[x, y] == 1:
					cellStrList.append('B')
				elif board[x, y] == 2:
					cellStrList.append('W')
			lines.append('|  ' + str(y)  + '  |  ' +  '  |  '.join( cellStrList ) + '  |')

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
