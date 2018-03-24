# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np


def evaluate_policy(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Evaluate the value of a policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """

    values = np.zeros(env.nS)
    i = 0
    while True:
        i += 1
        delta = 0
        for s in range(env.nS):
            old_v = values[s]
            v = 0
            a = policy[s]
            for (prob, next_state, reward, is_terminal) in env.P[s][a]:
                v += prob * (reward + gamma * values[next_state])
            values[s] = v
            delta = max(delta, abs(old_v - values[s]))

        if delta < tol:
            break
        if i > max_iterations:
            break
    return values, i


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """

    policy = np.zeros(env.nS, dtype='int')
    for s in range(env.nS):
        actions = []
        for a in range(env.nA):
            v = 0
            for (prob, next_state, reward, is_terminal) in env.P[s][a]:
                v += prob * (reward + gamma * value_function[next_state])
            actions.append(v)
        policy[s] = np.argmax(actions)
    return policy


def get_max_action(env, state, gamma, value_func):
    action_value = []
    for a in range(env.nA):
        v = 0
        for (prob, next_state, reward, is_terminal) in env.P[state][a]:
            v += prob * (reward + gamma * value_func[next_state])
        action_value.append(v)

    return np.argmax(action_value)

def improve_policy(env, gamma, value_func, policy):
    """Given a policy and value function improve the policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """

    policy_stable = True
    for s in range(env.nS):
        old_action = policy[s]
        action = get_max_action(env, s, gamma, value_func)
        if old_action != action:
            policy[s] = action
            policy_stable = False
    return policy_stable, policy


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    You should use the improve_policy and evaluate_policy methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    iteration = 0
    num_value_iter = 0
    while True:
        iteration += 1
        value_func, i = evaluate_policy(env, gamma, policy)
        num_value_iter += i
        policy_stable, policy = improve_policy(env, gamma, value_func, policy)
        if policy_stable:
            break
        if iteration > max_iterations:
            break
    return policy, value_func, iteration, num_value_iter


def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    return np.zeros(env.nS), 0


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)
