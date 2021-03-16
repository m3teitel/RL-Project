#!/usr/bin/env python3

# make sure to import the rom first
# `python3 -m retro.import "./Rom NoIntro/"`

import retro
import sys

def main(args):
    env = retro.make(game='MegaMan-Nes')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            return 0
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
