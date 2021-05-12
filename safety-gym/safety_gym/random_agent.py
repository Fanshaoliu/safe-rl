#!/usr/bin/env python

import argparse
import gym
import safety_gym  # noqa
import numpy as np  # noqa
from safe_rl.pg.buffer import CPOBuffer

def run_random(env_name):
    env = gym.make(env_name)
    obs = env.reset()
    done = False
    ep_ret = 0
    ep_cost = 0

    num_step = 0
    last_action = env.action_space.sample()
    while True:
        if done:
            print('Episode Return: %.3f \t Episode Cost: %.3f \t Episode num_step: %.3f'%(ep_ret, ep_cost, num_step))
            ep_ret, ep_cost = 0, 0
            obs = env.reset()
            num_step = 0
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()

        act = 0.5 * act + 0.5 * last_action
        last_action = act

        assert env.action_space.contains(act)
        obs, reward, done, info = env.step(act)

        num_step += 1
        # print('reward', reward)
        ep_ret += reward
        ep_cost += info.get('cost', 0)
        env.render()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Safexp-PointGoal1-v0')
    # parser.add_argument('--env', default='Safexp-CarGoal1-v0')
    # parser.add_argument('--env', default='doggogoal1-v0')

    args = parser.parse_args()
    run_random(args.env)