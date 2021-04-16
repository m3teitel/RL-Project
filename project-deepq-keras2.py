#!/usr/bin/python3

import sys
import os

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import keras
from keras import backend as K
from keras.models import Sequential
from keras import layers

import rl

import gym
import retro

class Network(keras.Model):
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
		
		return h


def main(args):
	env = retro.make('MegaMan-Nes', #5000, 0.9,
		#obs_type=retro.Observations.RAM,
		use_restricted_actions=retro.Actions.DISCRETE)
	
	
	return 0

if __name__ == '__main__':
	sys.exit(main(sys.argv))
