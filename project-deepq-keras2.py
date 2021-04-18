#!/usr/bin/env python3

# https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_cartpole.py
# use plaidml-setup to select target training device

import sys
import os

import gym
import retro

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import keras
from keras import backend as K
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam

#import rl
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

'''class Network(keras.Sequential):
	_n_features = 2048
	
	def __init__(self, input_shape, output_len):
		super(Network, self).__init__()
		
		self.l1 = layers.Conv2D(32, 8, strides=4, activation='relu')
		self.l2 = layers.Conv2D(64, 4, strides=2, activation='relu')
		self.l3 = layers.Conv2D(64, 3, strides=1, activation='relu')
		
		self.l4 = layers.Flatten()
		
		self.l5 = layers.Dense(self._n_features, activation='relu')
		self.l6 = layers.Dense(output_len, activation='linear')
	
	def call(self, inputs, training=False):
		debug = False
		
		if debug: print('input: ', inputs, file=sys.stderr)
		h = self.l1(inputs)
		if debug: print('layer 1: ', h, file=sys.stderr)
		h = self.l2(h)
		if debug: print('layer 2: ', h, file=sys.stderr)
		h = self.l3(h)
		if debug: print('layer 3: ', h, file=sys.stderr)
		h = self.l4(h)
		if debug: print('layer 4: ', h, file=sys.stderr)
		h = self.l5(h)
		if debug: print('layer 5: ', h, file=sys.stderr)
		h = self.l6(h)
		if debug: print('output: ', h, file=sys.stderr)
		
		return h'''


def main(args):
	env = retro.make('MegaMan-Nes', #5000, 0.9,
		#obs_type=retro.Observations.RAM,
		use_restricted_actions=retro.Actions.DISCRETE)
	#network = Network(env.observation_space.shape, env.action_space.n)
	'''network = Sequential()
	network.add(layers.Input(env.observation_space.shape))
	network.add(layers.Conv2D(32, 8, strides=4))#, activation='relu'))
	network.add(layers.Activation('relu'))
	network.add(layers.Conv2D(64, 4, strides=2))#, activation='relu'))
	network.add(layers.Activation('relu'))
	network.add(layers.Conv2D(64, 3, strides=1))#, activation='relu'))
	network.add(layers.Activation('relu'))
	network.add(layers.Flatten())
	network.add(layers.Dense(2048, activation='relu'))
	network.add(layers.Activation('relu'))
	network.add(layers.Dense(env.action_space.n))#, activation='relu'))
	network.add(layers.Activation('linear'))'''
	
	network = Sequential([
		layers.Conv2D(32, 8, strides=4, input_shape=env.observation_space.shape,
			activation='relu'),
		layers.Conv2D(64, 4, strides=3, activation='relu'),
		layers.Conv2D(64, 3, strides=2, activation='relu'),
		layers.Flatten(),
		layers.Dense(2048, activation='relu'),
		layers.Dense(env.action_space.n)])
	network.summary()
	print(env.observation_space.shape, file=sys.stderr)
	print('MODEL OUTPUT: ', network.output, file=sys.stderr)
	print('MODEL HAS LEN: ', hasattr(network.output, '__len__'), file=sys.stderr)
	#print('MODEL LEN: ', len(network.output), file=sys.stderr)
	
	memory = SequentialMemory(limit=50000, window_length=1)
	policy = EpsGreedyQPolicy(eps=0.1)
	dqn = DQNAgent(model=network, nb_actions=env.action_space.n,
		memory=memory, nb_steps_warmup=10, batch_size=32, policy=policy)
	dqn.compile(Adam(lr=1e-3), metrics=['mae'])
	
	dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)
	
	dqn.test(env, nb_episodes=10, visualize=True)
	
	
	
	
	return 0

if __name__ == '__main__':
	sys.exit(main(sys.argv))
