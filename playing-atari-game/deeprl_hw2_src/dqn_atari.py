#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse

import keras

import os
import gym
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Dropout)
from keras.models import Sequential
from keras.optimizers import Adam
from deeprl_hw2 import preprocessors, policy, core, objectives
from deeprl_hw2.dqn import DQNAgent

# Set the GPU storage usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def create_model(window, input_shape, num_actions,
                 model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understand your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    batch_size = 32
    model = Sequential()

    # first layer
    with tf.name_scope('hidden1'):
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(input_shape[0], input_shape[1], window)))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

    # second layer
    with tf.name_scope('hidden2'):
        model.add(Convolution2D(64, 4, 4, subsample=(4, 4)))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

    # third layer
    with tf.name_scope('hidden3'):
        model.add(Convolution2D(64, 3, 3, subsample=(4, 4)))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

    # forth layer
    with tf.name_scope('hidden4'):
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(num_actions))

    return model


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    try:
        os.makedirs(parent_dir)
    except OSError:
        pass
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main():  # noqa: D103

    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='Breakout-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('-ni', '--num_iterations', default=10, type=int, help='Num of iterations for training')
    parser.add_argument('-m', '--max_episode_length', default=60, type=int, help='Max episode length of a sequence')
    parser.add_argument('-ne', '--num_episodes', default=10, type=int, help='Num of epsidoes for evaluating')
    parser.add_argument('-r', '--replay_memory', default=10, type=int, help='The size of replay memory')
    parser.add_argument('-gamma', '--discount_factor', default=0.99, type=float, help='Discount factor of MDP')
    parser.add_argument('-ge', '--Greedy_epsilon', default=0.95, type=float, help='The probability to choose a greedy action')

    args = parser.parse_args()

    #args.input_shape = tuple(args.input_shape)

    args.output = get_output_folder(args.output, args.env)

    # the dirs to store results
    os.makedirs(args.output)
    os.chdir(args.output)

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

    env = gym.make('Breakout-v0')


    # Preprocess image
    preprocess_network = preprocessors.PreprocessorSequence('network')
    preprocess_memory = preprocessors.PreprocessorSequence('memory')

    # Policy choose
    Greedy = policy.GreedyEpsilonPolicy(0.95)
    DG = policy.LinearDecayGreedyEpsilonPolicy('attr_name', 1, 0.1, 1000000)

    # Create model from Atari paper
    model = create_model(window=4, input_shape=(84, 84), num_actions=6)

    # load weights
    location = '/'

    # Define tensorboard
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

    # Optimazor
    optimizor = Adam(lr=0.00025)

    # Create memory
    memory = core.ReplayMemory(max_size=args.replay_memory, phi_length=4, window_height=84, window_length=84, rng=np.random.RandomState(100))
    agent = DQNAgent(q_network=model, target=model, preprocessor={'network': preprocess_network, 'memory': preprocess_memory},
        memory=memory, policy={'Greedy': Greedy, 'DG': DG}, gamma=args.discount_factor, target_update_freq=100000, num_burn_in=args.replay_memory, train_freq=4, batch_size=32
        ,callbacks=tensorboard)
    agent.compile(optimizer= optimizor, loss_func=objectives.mean_huber_loss)
    agent.init_memory(env=env, max_episode_length=30)
    agent.fit(env=env, num_iterations=args.num_iterations, max_episode_length=args.max_episode_length)
    agent.evaluate(env=env, num_episodes=args.num_episodes, max_episode_length=args.max_episode_length)

    # store the hyperameters
    file_abs = "./hypermeters"
    with open(file_abs, "w") as f:
        f.write("Num of iterations:")
        f.write(str(args.num_iterations) + '\n')
        f.write("Max epsidoe length:")
        f.write(str(args.max_episode_length) + '\n')
        f.write("Num of episodes:")
        f.write(str(args.num_episodes) + '\n')
        f.write("Replay memory:")
        f.write(str(args.replay_memory) + '\n')
        f.write("Discount factor:")
        f.write(str(args.discount_factor) + '\n')


if __name__ == '__main__':
    main()
