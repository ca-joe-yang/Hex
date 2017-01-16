import numpy as np
from agent import *
from hex import HexEnv
import sys

N = sys.argv[1]
episodeNum = sys.argv[2]

timeout = 30
env = HexEnv(N, True)

blackData = 'data/Hex_'+str(N)+'x'+str(N)+'_Black.pkl'
whiteData = 'data/Hex_'+str(N)+'x'+str(N)+'_White.pkl'

#blackAgent = AlphaBetaSearchAgent(1)
blackAgent = MonteCarloSearchAgent(1, filename=blackData, time=timeout, mode='train')
#blackAgent = HumanAgent(1)
#blackAgent = RandomAgent(1)
#whiteAgent = HumanAgent(2)
whiteAgent = OnlyAttackAgent(2)
#whiteAgent = MonteCarloSearchAgent(2, filename=whiteData, time=timeout, mode='train')
#whiteAgent = ReflexAgent(2)
#whiteAgent = BetterRandomAgent(2)
env.setPlayerAgent(1, blackAgent)
env.setPlayerAgent(2, whiteAgent)

winCount = { 1: 0, 2: 0, 0: 0 }
for i in range(episodeNum):
	env.reset()
	env.autoPlay()
	winner = env.getWinner()
	winCount[winner] += 1
	print('Winner ', winner)

print('Player 1 win: ', winCount[1], ' / ', episodeNum)
print('Player 2 win: ', winCount[2], ' / ', episodeNum)
print('        Draw: ', winCount[0], ' / ', episodeNum)
