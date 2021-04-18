#!/usr/bin/env python3

import sys

import retro
import tensorflow as tf
import numpy as np

from tf_agents.environments import utils
from tf_agents.environments import GymWrapper


def main(args):
	gym_env = retro.make(game='MegaMan-Nes', #5000, 0.9,
		obs_type=retro.Observations.RAM,
		use_restricted_actions=retro.Actions.DISCRETE)
	env = GymWrapper(gym_env)
	utils.validate_py_environment(env, episodes=5)
	
	pass

if __name__ == '__main__':
	sys.exit(main(sys.argv))
