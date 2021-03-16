#!/usr/bin/env python3

# make sure to import the rom first
# `python3 -m retro.import "./Rom NoIntro/"`

import numpy as np
import retro
import sys

def main(args):
    env = retro.make(game='MegaMan-Nes')
    obs = env.reset()
#    print(obs, file=sys.stderr)
    # action[0] is B button, fire
    # action[1] is ? button, no action
    # action[2] is select button, no action
    # action[3] is ? button, no action
    # action[4] is ? button, no action
    # action[5] is ? button, no action
    # action[6] is left button, move left
    # action[7] is right button, move right
    # action[8] is A button, jump
    # ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
    # ['fire', None, 'no action', 'pause', 'climb ladder', 'descend ladder', 'move left', 'move right', 'jump']
    action = np.zeros(9)
    while True:
#        action = env.action_space.sample()
        print(action, file=sys.stderr)
#        print(env.buttons, file=sys.stderr)

        # obs looks like it might be the screen buffer?
        # rew ?
        # done boolean set if the level is completed or levels are out probably
        # info fetches the state of the predefined variables
        obs, rew, done, info = env.step(action)
#        print(obs.size, file=sys.stderr)
#        print(info, file=sys.stderr)
        print(rew, file=sys.stderr)
        env.render()
        if done:
            # end of episode
            return 0
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
