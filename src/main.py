import numpy as np
import agent
from hex import HexEnv

episodeNum = 10

env = HexEnv(1, False)

env.setPlayerAgent(1, agent.RandomAgent)
env.setPlayerAgent(2, agent.BetterRandomAgent)

winCount = { 1: 0, 2: 0, 0: 0 }
for i in range(episodeNum):
	env.reset(5)
	result = env.autoPlay()
	winCount[result] += 1

print('Player 1 win: ', winCount[1], ' / ', episodeNum)
print('Player 2 win: ', winCount[2], ' / ', episodeNum)
print('        Draw: ', winCount[0], ' / ', episodeNum)