import numpy as np
from agent import *
from hex import HexEnv

episodeNum = 1

env = HexEnv(5, True)

env.setPlayerAgent(1, UCTAgent())
env.setPlayerAgent(2, BetterRandomAgent())

winCount = { 1: 0, 2: 0, 0: 0 }
for i in range(episodeNum):
	env.reset()
	env.autoPlay()
	winner = env.getWinner()
	winCount[winner] += 1

print('Player 1 win: ', winCount[1], ' / ', episodeNum)
print('Player 2 win: ', winCount[2], ' / ', episodeNum)
print('        Draw: ', winCount[0], ' / ', episodeNum)
