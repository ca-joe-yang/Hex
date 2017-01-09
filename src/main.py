import numpy as np
import agent
from hex import HexEnv

episodeNum = 100

env = HexEnv(6, False)

env.setPlayerAgent(1, agent.RandomAgent)
env.setPlayerAgent(2, agent.MonteCarloTreeSearchAgent)

winCount = { 1: 0, 2: 0, 0: 0 }
for i in range(episodeNum):
	env.reset()
	env.autoPlay()
	winner = env.getWinner()
	winCount[winner] += 1

print('Player 1 win: ', winCount[1], ' / ', episodeNum)
print('Player 2 win: ', winCount[2], ' / ', episodeNum)
print('        Draw: ', winCount[0], ' / ', episodeNum)
