#!/usr/bin/env python3

# make sure to import the rom first
# `python3 -m retro.import "./Rom NoIntro/"`

#import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import retro
import sys
import multiprocessing

from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.policy import EpsGreedy
#from mushroom_rl.algorithms.value.td import SARSA
from mushroom_rl.core.core import Core

from mushroom_rl.algorithms.value import DQN
from mushroom_rl.approximators.parametric import TorchApproximator

from gym import spaces as gym_spaces
from mushroom_rl.environments import MDPInfo
from mushroom_rl.utils.spaces import *

from util import RetroEnvironment

from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.utils.table import Table

class RetroFullEnvironment(RetroEnvironment):
	def __init__(self, game, horizon, gamma, **env_args):
		super.__init__(self, game, horizon, gamma, **env_args)
		
	def step(self, action):
		action = self._convert_action(action)
		obs, rew, done, info = self.env.step(action)
		return obs, rew, done, info

# This network does not work
# Dimension error, need to rework
class Network(nn.Module):
	n_features = 2048
	def __init__(self, input_shape, output_shape, **kwargs):
		super(Network, self).__init__()
		
		n_input = input_shape[0]
		n_output = output_shape[0]

		self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=4)
		self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		self._h4 = nn.Linear(3136, self.n_features)
		self._h5 = nn.Linear(self.n_features, n_output)

		nn.init.xavier_uniform_(self._h1.weight,
								gain=nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self._h2.weight,
								gain=nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self._h3.weight,
								gain=nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self._h4.weight,
								gain=nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self._h5.weight,
								gain=nn.init.calculate_gain('linear'))
	
	def forward(self, state, action=None):
		print(state, file=sys.stderr)
		print(state.size(), file=sys.stderr)
		h = F.relu(self._h1(state.float() / 255.))
		h = F.relu(self._h2(h))
		h = F.relu(self._h3(h))
		h = F.relu(self._h4(h.view(-1, 3136)))
		q = self._h5(h)

		if action is None:
			return q
		else:
			q_acted = torch.squeeze(q.gather(1, action.long()))

			return q_acted

def main(argv):
	#env = retro.make(game='MegaMan-Nes', obs_type=retro.Observations.RAM)
	mdp = RetroEnvironment('MegaMan-Nes', 5000, 0.9,
		obs_type=retro.Observations.RAM,
		use_restricted_actions=retro.Actions.DISCRETE)
	
	epsilon = Parameter(value=1.)
	learning_rate = Parameter(value=0.3)
	
	train_frequency = 4
	evaluation_frequency = 250000
	target_update_frequency = 10000
	initial_replay_size = 50000
	max_replay_size = 500000
	test_samples = 125000
	max_steps = 50000000
	
	policy = EpsGreedy(epsilon=epsilon)
	#agent = SARSA(env.info, policy, learning_rate)
	#agent = CustomSARSA(env.info, policy, learning_rate)
	
	#history_length = 1
	
	
	optimizer = {
		'class': optim.Adam,
		'params': dict(lr=0.00025)
	}

	approximator = TorchApproximator
	approximator_params = dict(
		network=Network,
		input_shape=mdp.info.observation_space.shape,
		output_shape=(mdp.info.action_space.n,),
		n_actions=mdp.info.action_space.n,
		n_features=Network.n_features,
		optimizer=optimizer,
		loss=F.smooth_l1_loss
	)
	
	algorithm_params = dict(
		batch_size=32,
		target_update_frequency=target_update_frequency // train_frequency,
		replay_memory=None,
		initial_replay_size=initial_replay_size,
		max_replay_size=max_replay_size
	)
	agent = DQN(mdp.info, policy, approximator,
		approximator_params=approximator_params, **algorithm_params)
	
	core = Core(agent, mdp)
	core.learn(n_steps=1000000, n_steps_per_fit=1)
	core.evaluate(n_episodes=10, render=True)
	
	
	'''print(agent.Q.shape, file=sys.stderr)
	shape = agent.Q.shape
	q = np.zeros(shape)
	for i in range(shape[0]):
		for j in range(shape[1]):
			state = np.array([i])
			action = np.array([j])
			q[i, j] = agent.Q.predict(state, action)
	print(q)'''
	
	return 0

if __name__ == '__main__':
	sys.exit(main(sys.argv))
