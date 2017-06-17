#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input
import matplotlib.pyplot as plt
import deeprl_hw1.lake_envs as lake_env
import gym
from deeprl_hw1.rl import *


def print_env_info(env):
    print('Environment has %d states and %d actions.' % (env.nS, env.nA))


def print_model_info(env, state, action):
    transition_table_row = env.P[state][action]
    print(
        ('According to transition function, '
         'taking action %s(%d) in state %d leads to'
         ' %d possible outcomes') % (lake_env.action_names[action],
                                     action, state, len(transition_table_row)))
    for prob, nextstate, reward, is_terminal in transition_table_row:
        state_type = 'terminal' if is_terminal else 'non-terminal'
        print(
            '\tTransitioning to %s state %d with probability %f and reward %f'
            % (state_type, nextstate, prob, reward))


def main():
    # create the environment
    # env = gym.make('FrozenLake-v0')
    # uncomment next line to try the deterministic version
    env = gym.make('Stochastic-4x4-FrozenLake-v0')

    print_env_info(env)
    print_model_info(env, 0, lake_env.DOWN)
    print_model_info(env, 1, lake_env.DOWN)
    print_model_info(env, 14, lake_env.RIGHT)

    input('Hit enter to run a policy...')

    gamma = 0.9

    policy, value_func, improve_steps, value_steps = \
        policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3)
    print('Agent took %d value steps & %d improve steps to reache the optimal value function'
          % (value_steps, improve_steps))
    print("The optimal policy is:")
    print_policy(policy, action_names={0: 'L', 1: 'D', 2: 'R', 3: 'U'})
    value_matrix = value_func.reshape((-1, 4))
    print(value_matrix)
    fig1 = plt.imshow(value_matrix)
    plt.colorbar(fig1, orientation='vertical')
    plt.show()

    value_func, num_steps = value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3)
    policy = value_function_to_policy(env, gamma, value_func)

    print('Agent took %d steps to reach the optimal value function'
          % num_steps)
    print("The optimal policy is:")
    print_policy(policy, action_names={0: 'L', 1: 'D', 2: 'R', 3: 'U'})
    value_matrix = value_func.reshape((-1, 4))
    print(value_matrix)
    fig2 = plt.imshow(value_matrix)
    plt.colorbar(fig2, orientation='vertical')
    plt.show()


if __name__ == '__main__':
    main()
