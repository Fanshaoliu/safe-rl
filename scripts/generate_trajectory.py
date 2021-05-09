#!/usr/bin/env python

import time
import numpy as np
from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger



def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    ts = []  # tragectory_saver
    fts = []  # final tragectory_saver


    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :("

    logger = EpochLogger()
    s, r, d, ep_ret, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(s)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        s_, r, d, info = env.step(a)

        # c_ = []
        # for i,v in info.items():
        #     c_.append(v)
        # print(info)
        ts.append([s, a, r, info['cost'], s_])

        s = s_

        ep_ret += r
        ep_cost += info.get('cost', 0)
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d'%(n, ep_ret, ep_cost, ep_len))
            s, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
            n += 1
            fts.append(ts)
            ts = []

    np.save("tragectory.npy", np.array(fts))

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpCost', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default="/Users/liushaofan/PycharmProjects/safety-starter-agents/data/2021-04-11_ppo_lagrangian_PointGoal1v0/2021-04-11_20-37-00-ppo_lagrangian_PointGoal1v0_s0")
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=10)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action, sess = load_policy(args.fpath,
                                        args.itr if args.itr >=0 else 'last',
                                        args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))
