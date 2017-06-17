__author__ = 'haoxu'
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input
import matplotlib.pyplot as plt
import deeprl_hw1.lake_envs as lake_env
import gym
import numpy as np
import time

def choose_action(state):


def run_SARSA_policy(env):
    """ Run the SARSA policy for the given environment
    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    max_episodes = 2000
    max_iterations = 1e2
    total_reward = 0
    total_steps = 0
    for episode in range(max_episodes):
        initial_state = env.reset()
        policy = np.zeros(env.nS, dtype=int)
        action = choose_action(initial_state)
        for iter in range(max_iterations):


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

    total_reward, num_steps = run_SARSA_policy(env)


if __name__ == '__main__':
    main()
