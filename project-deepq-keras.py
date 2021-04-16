#!/usr/bin/python3

import sys
import os

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import numpy as np
import keras
from keras import backend as K
#from tensorflow import keras
#import keras.applications as kapp
from keras.models import Sequential
#from tensorflow.keras import layers
from keras import layers

import gym
import retro

# ~ class EnvironmentWrapper(object):
	# ~ def __init__(self, env):
		# ~ self.env = env
		
		# ~ self.observation_space = None
		# ~ if (obs_shape == None):
			# ~ self.observation_space = self._convert_gym_space(self.env.observation_space)
		# ~ else:
			# ~ self.observation_space = Discrete(obs_shape)
		# ~ self.action_space = self._convert_gym_space(self.env.action_space)
	
	# ~ @staticmethod
	# ~ def _convert_gym_space(space):
		# ~ if isinstance(space, gym_spaces.Discrete):
			# ~ return Discrete(space.n)
		# ~ elif isinstance(space, gym_spaces.Box):
			# ~ return Box(low=space.low, high=space.high, shape=space.shape)
		# ~ else:
			# ~ raise ValueError

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

# Constants
def EPSILON_MAX(): return 1.0
def EPSILON_MIN(): return 0.1
def EPSILON_INTERVAL(): return (EPSILON_MAX() - EPSILON_MIN())
def BATCH_SIZE(): return 32
def MAX_STEPS_PER_EPISODE(): return 10000

def EPSILON_RANDOM_FRAMES(): return 50000
def UPDATE_AFTER_ACTIONS(): return 4
def UPDATE_TARGET_NETWORK(): return 10000
def MAX_MEMORY_LENGTH(): return 100000

def main(args):
	env = retro.make('MegaMan-Nes', #obs_type=retro.Observations.RAM,
		use_restricted_actions=retro.Actions.DISCRETE)
	#envWrapper
	
	epsilon = EPSILON_MAX()
	gamma = 0.99
	model = Network(env.observation_space.shape, env.action_space.n)
	target = Network(env.observation_space.shape, env.action_space.n)
	optimizer = keras.optimizers.Adam(lr=0.00025, clipnorm=1.0)
	
	action_history = []
	state_history = []
	state_next_history = []
	rewards_history = []
	done_history = []
	episode_reward_history = []
	running_reward = 0
	episode_count = 0
	frame_count = 0
	
	epsilon_greedy_frames = 1000000.0
	
	while True:
		state = np.array(env.reset())
		episode_reward = 0
		
		for t in range(1, MAX_STEPS_PER_EPISODE()):
			env.render()
			frame_count += 1
			
			if frame_count < EPSILON_RANDOM_FRAMES() or epsilon > np.random.rand():
				action = np.random.choice(env.action_space.n)
			else:
				state_tensor = K.constant(state)
				state_tensor = K.expand_dims(state_tensor, 0)
				action_probs = model(state_tensor, training=False)
				action = tf.argmax(action_probs[0]).numpy()
			
			epsilon -= EPSILON_INTERVAL() / epsilon_greedy_frames
			epsilon = max(epsilon, EPSILON_MIN())
			
			state_next, reward, done, _info = env.step(action)
			state_next = K.constant(state_next)
			
			episode_reward += reward
			
			action_history.append(action)
			state_history.append(state)
			state_next_history.append(state_next)
			done_history.append(done)
			rewards_history.append(reward)
			state = state_next
			
			if frame_count % UPDATE_AFTER_ACTIONS() == 0 and len(done_history) > BATCH_SIZE():
				indices = np.random.choice(range(len(done_history)), size=BATCH_SIZE())
				print([state_history[i] for i in indices], file=sys.stderr)
				#state_sample = np.array([state_history[i] for i in indices])
				state_sample = K.constant([state_history[i] for i in indices])
				#state_next_sample = np.array([state_next_history[i] for i in indices])
				state_next_sample = K.constant([state_next_history[i] for i in indices])
				rewards_sample = [rewards_history[i] for i in indices]
				action_sample = [action_history[i] for i in indices]
				done_sample = K.constant([float(done_history[i]) for i in indices])
				
				future_rewards = target.predict(state_next_sample)
				updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)
				
				masks = K.one_hot(action_sample, env.action_space.n)
				
				# train model
				
				#grads = 
				
			if frame_count % UPDATE_TARGET_NETWORK() == 0:
				model_target.set_weights(model.get_weights)
				print('running reward: {:.2f} at episode {}, frame count {}'
					.format(running_reward, episode_count, frame_count))
			
			if len(rewards_history) > MAX_MEMORY_LENGTH():
				del rewards_history[:1]
				del state_history[:1]
				del state_next_history[:1]
				del action_history[:1]
				del done_history[:1]
			
			if done:
				break
				
		episode_reward_history.append(episode_reward)
		if len(episode_reward_history) > 100:
			del episode_reward_history[:1]
		running_reward = np.mean(episode_reward_history)

		episode_count += 1
    
    
if __name__ == '__main__':
	sys.exit(main(sys.argv))
