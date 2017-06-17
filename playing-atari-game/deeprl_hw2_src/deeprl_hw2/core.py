"""Core classes."""
from deeprl_hw2 import preprocessors
import numpy as np

class Sample:
    """Represents a reinforcement learning sample.

    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.

    Note: This is not the most efficient way to store things in the
    replay memory, but it is a convenient class to work with when
    sampling batches, or saving and loading samples while debugging.

    Parameters
    ----------
    state: array-like
      Represents the state of the MDP before taking an action. In most
      cases this will be a numpy array.
    action: int, float, tuple
      For discrete action domains this will be an integer. For
      continuous action domains this will be a floating point
      number. For a parameterized action MDP this will be a tuple
      containing the action and its associated parameters.
    reward: float
      The reward received for executing the given action in the given
      state and transitioning to the resulting state.
    next_state: array-like
      This is the state the agent transitions to after executing the
      `action` in `state`. Expected to be the same type/dimensions as
      the state.
    is_terminal: boolean
      True if this action finished the episode. False otherwise.
    """
    def __init__(self, s, a, r, next_state, terminal):
        self.sample = [s, a, r, next_state, terminal]

class ReplayMemory:
    """Interface for replay memories.

    We have found this to be a useful interface for the replay
    memory. Feel free to add, modify or delete methods/attributes to
    this class.

    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.

    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just randomly draw samples saved in your memory).

    However, the above approach will waste a lot of memory (as states
    will be stored multiple times in s as next state and then s' as
    state, etc.). Depending on your machine resources you may want to
    implement a version that stores samples in a more memory efficient
    manner.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, max_size, window_length, window_height, phi_length, rng):
        """Setup memory.

        You should specify the maximum size of the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        self.max_size = max_size

        self.width = window_length
        self.height = window_height
        self.length = phi_length
        self.rng = rng

        self.bottom = 0
        self.top = 0
        self.size = 0

        self.states = np.zeros((self.max_size, self.height, self.width), dtype='uint8')
        self.actions = np.zeros(max_size, dtype='int32')
        self.rewards = np.zeros(max_size, dtype='float32')
        self.terminal = np.zeros(max_size, dtype='bool')


    def __len__(self):
        return max(0, self.size - self.length)

    def append(self, state, action, reward, terminal):
        preprocessor = preprocessors.Preprocessor()
        state_preprocessed = preprocessor.process_state_for_memory(state)
        reward_preprocessed = preprocessor.process_reward(reward)

        self.states[self.top] = state_preprocessed
        self.actions[self.top] = action
        self.rewards[self.top] = reward_preprocessed
        self.terminal[self.top] = terminal

        if self.size == self.max_size:
            self.bottom = (self.bottom + 1) % self.max_size
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_size

    def end_episode(self, final_state, is_terminal):
        self.final_state = final_state
        self.is_terminal = is_terminal

    def sample(self, batch_size, indexes=None):
        batch = []
        count = 0
        while count < batch_size:
            index = self.rng.randint(self.bottom, self.bottom + self.size - self.length)

            all_indices = np.arange(index, index + self.length + 1)
            state_indices = all_indices[:-1]
            next_state_indices = all_indices[1:]
            end_index = index + self.length - 1

            # Check that the initial state corresponds entirely to a
            # single episode, meaning none but its last frame (the
            # second-to-last frame in imgs) may be terminal. If the last
            # frame of the initial state is terminal, then the last
            # frame of the transitioned state will actually be the first
            # frame of a new episode, which the Q learner recognizes and
            # handles correctly during training by zeroing the
            # discounted future reward estimate.
            if np.any(self.terminal.take(all_indices[0:-1], mode='wrap')):
                continue

            # Add the state transition to the response.
            state = self.states.take(state_indices, axis=0, mode='wrap').reshape(1, self.height, self.width, self.length)
            next_state = self.states.take(next_state_indices, axis=0, mode='wrap').reshape(1, self.height, self.width, self.length)
            reward = self.rewards.take(end_index, mode='wrap')
            action = self.actions.take(end_index, mode='wrap')
            is_terminal = self.terminal.take(end_index, mode='wrap')

            # sample add to memory
            sample = Sample(state, action, reward, next_state, is_terminal)
            batch.append(sample)
            count += 1

        preprocess_batch = preprocessors.Preprocessor()
        batch = preprocess_batch.process_batch(batch)
        return batch

    def clear(self):
        self.states = np.zeros((self.max_size, self.height, self.width), dtype='uint8')
        self.actions = np.zeros(self.max_size, dtype='int32')
        self.rewards = np.zeros(self.max_size, dtype='float32')
        self.terminal = np.zeros(self.max_size, dtype='bool')
        # to record the size and top and bottom of the ring buffer
        self.bottom = 0
        self.top = 0
        self.size = 0
