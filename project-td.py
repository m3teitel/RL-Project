#!/usr/bin/env python3

# make sure to import the rom first
# `python3 -m retro.import "./Rom NoIntro/"`

import numpy as np
import retro
import sys
import multiprocessing

from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.algorithms.value.td import SARSA
from mushroom_rl.core.core import Core

from gym import spaces as gym_spaces
from mushroom_rl.environments import Environment, MDPInfo
from mushroom_rl.utils.spaces import *

def SCREEN_ADDR(): return 0x460     # current screen number
def MMX_ADDR(): return 0x480        # x-position of megaman
def MMY_ADDR(): return 0x600        # y-position of megaman

from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.utils.table import Table

'''class CustomSARSA(TD):
	def __init__(self, mdp_info, policy, learning_rate):
		Q = Table(mdp_info.size, dtype=np.float16)
		super().__init__(mdp_info, policy, Q, learning_rate)
	
	def _update(self, state, action, reward, next_state, absorbing):
		q_current = self.Q[state, action]

		self.next_action = self.draw_action(next_state)
		q_next = self.Q[next_state, self.next_action] if not absorbing else 0.

		self.Q[state, action] = q_current + self.alpha(state, action) * (
			reward + self.mdp_info.gamma * q_next - q_current)'''

# derived from source code for mushroom_rl integration with Gym
class RetroEnvironment(Environment):
	def __init__(self, game, horizon, gamma, **env_args):
		self._close_at_stop = True
		
		self.env = retro.make(game=game, **env_args)
		
		self.env._max_episode_steps = np.inf
		
		action_space = self._convert_gym_space(self.env.action_space)
		#observation_space = self._convert_gym_space(self.env.observation_space)
		observation_space = Discrete(256*256*256)
		mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
	
		super().__init__(mdp_info)
	
	def reset(self, state):
		return self.env.reset()
	
	def step(self, action):
		action = self._convert_action(action)
		obs, rew, done, info = self.env.step(action)
		#if rew == 0:
		#	rew = -1
		
		#x = obs[MMX_ADDR()] * 256 + obs[MMX_ADDR()+1]
		#y = (obs[MMY_ADDR()] * 256 + obs[MMY_ADDR()+1]) & 0x3fff
		x = obs[MMX_ADDR()]
		y = obs[MMY_ADDR()]
		screen = obs[SCREEN_ADDR()]
		
		#i = obs[SCREEN_ADDR()] * 65536 + obs[MMY_ADDR()] * 256 + obs[MMX_ADDR()]
		#i = screen * 65536**2 + y * 65536 + x
		#i = y * 65536 + x
		i = screen * 65536 + y * 256 + x
		#i = screen * 16384 * 256 + y * 256 + x
		return i, rew, done, info
	
	def render(self):
		self.env.render()
	
	def stop(self):
		try:
			if self._close_at_stop:
				self.env.close
		except:
			pass
	
	def _convert_action(self, action):
		'''real_action = list(map(int, bin(action[0])[2:]))
		real_action = np.pad(real_action, (self.env.action_space.n - len(real_action), 0), 'constant')
		return real_action'''
		return action[0]
	
	@staticmethod
	def _convert_gym_space(space):
		if isinstance(space, gym_spaces.Discrete):
			return Discrete(space.n)
		elif isinstance(space, gym_spaces.Box):
			return Box(low=space.low, high=space.high, shape=space.shape)
		else:
			raise ValueError


def main(argv):
	#env = retro.make(game='MegaMan-Nes', obs_type=retro.Observations.RAM)
	env = RetroEnvironment('MegaMan-Nes', 5000, 0.9, obs_type=retro.Observations.RAM, \
		use_restricted_actions=retro.Actions.DISCRETE)
	
	epsilon = Parameter(value=1.)
	learning_rate = Parameter(value=0.3)
	
	policy = EpsGreedy(epsilon=epsilon)
	agent = SARSA(env.info, policy, learning_rate)
	#agent = CustomSARSA(env.info, policy, learning_rate)
	
	core = Core(agent, env)
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
