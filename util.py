from gym import spaces as gym_spaces
import retro
import numpy as np
from mushroom_rl.environments import Environment, MDPInfo
from mushroom_rl.utils.spaces import *

def SCREEN_ADDR(): return 0x460     # current screen number
def MMX_ADDR(): return 0x480        # x-position of megaman
def MMY_ADDR(): return 0x600        # y-position of megaman

# derived from source code for mushroom_rl integration with Gym
class RetroEnvironment(Environment):
	def __init__(self, game, horizon, gamma, obs_shape=None, **env_args):
		self._close_at_stop = True
		
		self.env = retro.make(game=game, **env_args)
		
		self.env._max_episode_steps = np.inf
		
		observation_space = None
		if (obs_shape == None):
			observation_space = self._convert_gym_space(self.env.observation_space)
		else:
			observation_space = Discrete(obs_shape)
		
		action_space = self._convert_gym_space(self.env.action_space)
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

class RetroFullEnvironment(RetroEnvironment):
	def __init__(self, game, horizon, gamma, **env_args):
		super(RetroFullEnvironment, self).__init__(game, horizon, gamma,
			**env_args)
		
	def step(self, action):
		action = self._convert_action(action)
		obs, rew, done, info = self.env.step(action)
		return obs, rew, done, info
