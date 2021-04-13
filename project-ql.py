#!/usr/bin/env python3

# make sure to import the rom first
# `python3 -m retro.import "./Rom NoIntro/"`

import numpy as np
import retro
import sys
import multiprocessing
from sklearn.ensemble import ExtraTreesRegressor

from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.algorithms.value.td import QLearning
from mushroom_rl.core.core import Core

from gym import spaces as gym_spaces
from mushroom_rl.environments import Environment, MDPInfo
from mushroom_rl.utils.spaces import *

from util import RetroEnvironment

def main(argv):
	#env = retro.make(game='MegaMan-Nes', obs_type=retro.Observations.RAM)
	env = RetroEnvironment('MegaMan-Nes', 10000, 0.9, obs_shape=256**3, \
		obs_type=retro.Observations.RAM, \
		use_restricted_actions=retro.Actions.DISCRETE)
	
	epsilon = Parameter(value=1.)
	learning_rate = Parameter(value=0.3)
	
	policy = EpsGreedy(epsilon=epsilon)

	agent = QLearning(env.info, policy, learning_rate)
	#agent = CustomSARSA(env.info, policy, learning_rate)
	
	core = Core(agent, env)
	core.learn(n_episodes=10, n_steps_per_fit=1)
	core.evaluate(n_episodes=2, render=True)
	
	
	# print(agent.Q.shape, file=sys.stderr)
	# shape = agent.Q.shape
	# q = np.zeros(shape)
	# for i in range(shape[0]):
	# 	for j in range(shape[1]):
	# 		state = np.array([i])
	# 		action = np.array([j])
	# 		q[i, j] = agent.Q.predict(state, action)
	# print(q)
	
	return 0

if __name__ == '__main__':
	sys.exit(main(sys.argv))
