"""Main DQN agent."""

import os
from deeprl_hw2.policy import *
from deeprl_hw2 import utils
import time

class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
                 q_network,
                 target,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 callbacks):
        self.q_network = q_network
        self.target = target
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in =num_burn_in
        self.train_freq = train_freq
        self.batch_size =batch_size
        self.callbacks = callbacks

    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """

        self.optimizer = optimizer
        self.loss_function = loss_func
        self.q_network.compile(optimizer=self.optimizer,loss=self.loss_function)
        self.target.compile(optimizer=self.optimizer, loss=self.loss_function)


    def calc_source_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """

        return self.q_network.predict(state)

    def calc_target_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """

        return self.target.predict(state)

    def select_action(self, state, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """

        q_values = self.calc_source_values(state)[0]
        if kwargs['is_train'] is True:
            policy_use = self.policy['DG']
            return policy_use.select_action(is_training=True, q_values=q_values, total_t=kwargs['total_t'])
        elif kwargs['is_train'] is False:
            policy_use = self.policy['Greedy']
            return policy_use.select_action(q_values = q_values)

    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """

        sample_batch = self.memory.sample(batch_size=self.batch_size)
        y_target_batch = np.empty([self.batch_size, 4])
        states = np.empty([self.batch_size, 84, 84, 4])
        count = 0
        for sample in sample_batch:
            state, action, reward, next_state, is_terminal = sample.sample
            if is_terminal:
                y_target_r = reward
                y_target_batch.append(y_target_r)
            else:
                # Using SARSA
                y_target_r = reward + self.gamma * np.amax(self.calc_target_values(next_state),axis=0)
                y_target_batch[count] = y_target_r
            states[count] = state
            count += 1
        self.q_network.fit(states, y_target_batch, callbacks=[self.callbacks])

    def init_memory(self,env, max_episode_length):
        state = env.reset()
        for j in range(max_episode_length):
            #print(state)
            state_process = self.preprocessor['network'].process(state)
            action = self.select_action(state_process, is_train=True, start_value=0.1, end_value=0.9,
                                        num_steps=max_episode_length,total_t=j)
            #print action
            next_state, reward, is_terminal, _ = env.step(action)
            img_process = self.preprocessor['memory'].process(state)
            self.memory.append(img_process, action, reward, is_terminal)
            state = next_state

    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        dir_name = 'weight-' + str(num_iterations) + 'episodes'
        os.makedirs(dir_name)
        state = env.reset()
        # process the state for choose action
        state_process = self.preprocessor['network'].process(state)
        for i in range(num_iterations):
            state = env.reset()
            # process the state for chosen action
            state_process = self.preprocessor['network'].process(state)
            for j in range(max_episode_length):
                action = self.select_action(state=state_process,is_train=True, start_value=0.1,
                                            end_value=0.9,num_steps=max_episode_length,total_t=j)
                next_state, reward, is_terminal, _ = env.step(action)
                state_memory = self.preprocessor['memory'].process(state)
                self.memory.append(state_memory, action, reward, is_terminal)
                state = next_state
                if j % self.train_freq == 0:
                    self.update_policy()
                if j % self.target_update_freq == 0:
                    utils.get_hard_target_model_updates(self.target, self.q_network)
        self.q_network.save_weights(dir_name+'/weight_episode')

    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """

        dir_name = 'results-' + str(num_episodes) + 'episodes'
        os.makedirs(dir_name)
        rewards = np.empty([num_episodes, max_episode_length])
        for i in range(num_episodes):
            state = env.reset()
            for j in range(max_episode_length):
                env.render()
                time.sleep(0.1)
                state_process = self.preprocessor['network'].process(state)
                action = self.select_action(state_process, is_train=False, epsilon=0.9)
                next_state, reward, is_terminal, _ = env.step(action)
                if is_terminal:
                    break
                rewards[i][j] = reward
                state = next_state

        # print the reward
        avg_reward_per_episode, sum_reward_per_episode, total_reward = np.average(rewards, axis=1), np.sum(rewards, axis=1), np.sum(rewards)
        print("Averge reward in every episode is ")
        print(avg_reward_per_episode)
        print("Total reward in every episode is ")
        print(sum_reward_per_episode)
        print("Total reward is ")
        print(total_reward)
        file_abs = dir_name + "/rewards"
        with open(file_abs, "w") as f:
            f.write("Averge reward per episode\n")
            f.write(str(avg_reward_per_episode) + '\n')
            f.write("Total reward per episode\n")
            f.write(str(sum_reward_per_episode)+ '\n')
            f.write("Total reward\n")
            f.write(str(total_reward))
