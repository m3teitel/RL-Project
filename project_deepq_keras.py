#!/usr/bin/env python3

# make sure to import the rom first
# `python3 -m retro.import "./Rom NoIntro/"`

import os
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

import keras
from keras import backend as K
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import mean_squared_error

import retro
import sys
import multiprocessing

from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.core.core import Core

from mushroom_rl.algorithms.value import DQN
from keras_approximator import KerasApproximator

from gym import spaces as gym_spaces
from mushroom_rl.environments import MDPInfo

from util import RetroFullEnvironment

def model(input_shape, output_shape, n_features=2048, print_summary=False):
	net = Sequential([
		layers.Conv2D(32, 8, strides=4, input_shape=input_shape,
			activation='relu'),
		layers.Conv2D(64, 4, strides=3, activation='relu'),
		layers.Conv2D(64, 3, strides=2, activation='relu'),
		layers.Flatten(),
		layers.Dense(n_features, activation='relu'),
		layers.Dense(output_shape[0])])
	if print_summary:
		net.summary()
	return net

def main(argv):
	mdp = RetroFullEnvironment('MegaMan-Nes', 5000, 0.9,
		#obs_type=retro.Observations.RAM,
		use_restricted_actions=retro.Actions.DISCRETE)
	
	epsilon = Parameter(value=1.)
	learning_rate = Parameter(value=0.3)
	
	train_frequency = 4
	evaluation_frequency = 250000
	target_update_frequency = 10000
	initial_replay_size = 50000
	#initial_replay_size = 500
	max_replay_size = 500000
	test_samples = 125000
	max_steps = 50000000
	
	policy = EpsGreedy(epsilon=epsilon)
	
	optimizer = {
		'class': Adam,
		'params': dict(lr=0.00025)
	}

	approximator = KerasApproximator
	approximator_params = dict(
		network=model,
		input_shape=mdp.info.observation_space.shape,
		output_shape=(mdp.info.action_space.n,),
		n_actions=mdp.info.action_space.n,
		n_features=2048,
		optimizer=optimizer,
		loss=mean_squared_error,
		print_summary=True
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
	#core.learn(n_steps=1000000, n_steps_per_fit=1)
	core.learn(n_steps=100000, n_steps_per_fit=1)
	core.evaluate(n_episodes=10, render=True)
	
	return 0

if __name__ == '__main__':
	sys.exit(main(sys.argv))
