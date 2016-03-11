'''Includes experiment definitions that can be used to test RC-NFQ learning in
the VisionRobot environment or to test Tabular Q-Learning in the GridWorld 
environment.
'''
import numpy as np
from environments import GridWorld
from rcnfq import NFQ


class ConvolutionalNFQExperiment:
    def __init__(self,
                 env,
                 nb_episodes,
                 max_steps_per_episode,
                 nb_samples=500,
                 sliding_window=5000,
                 target_network_update_freq=100,
                 lr=0.001):
        """Instantiate a Convolutional NFQ experiment.

        Parameters
        ----------
        environment : should expose the following methods and attributes:
            - state_dim
            - nb_actions
            - state
            - terminal()
            - act(action)
        """
        # ---------------------------- Initialization ----------------------------
        # Allocate NumPy arrays
        self.state = 'INITIALIZING'
        self.env = env
        self.num_episodes = nb_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.history_size = nb_episodes * max_steps_per_episode
        self.nb_samples = nb_samples
        self.sliding_window = sliding_window
        self.target_network_update_freq = target_network_update_freq
        self.lr = lr
        self.epsilon = 0.10

        self._init()

    def _init(self):
        print('Initializing experiment.')

        # Stores the data from all episodes
        self.D_s = np.zeros((self.history_size,
                             self.env.state_dim[0],
                             self.env.state_dim[1],
                             self.env.state_dim[2]),
                             dtype=np.float32)
        self.D_a = np.zeros(self.history_size, dtype=np.int32)
        self.D_r = np.zeros(self.history_size, dtype=np.float32)
        self.D_s_prime = np.zeros((self.history_size,
                                   self.env.state_dim[0],
                                   self.env.state_dim[1],
                                   self.env.state_dim[2]),
                                   dtype=np.float32)

        # Stores the data within each episode
        self.D_new_s = np.zeros((self.max_steps_per_episode,
                             self.env.state_dim[0],
                             self.env.state_dim[1],
                             self.env.state_dim[2]),
                             dtype=np.float32)
        self.D_new_a = np.zeros(self.max_steps_per_episode, dtype=np.int32)
        self.D_new_r = np.zeros(self.max_steps_per_episode, dtype=np.float32)
        self.D_new_s_prime = np.zeros((self.max_steps_per_episode,
                                   self.env.state_dim[0],
                                   self.env.state_dim[1],
                                   self.env.state_dim[2]),
                                   dtype=np.float32)

        # Stores the reward history
        self.r_history = np.zeros((self.history_size), dtype=np.float32)
        self.episode_r_history = np.zeros((self.num_episodes), dtype=np.float32)

        self.nfq = NFQ(state_dim=self.env.state_dim,
                       nb_actions=self.env.nb_actions,
                       terminal_states=None,
                       convolutional=True,
                       discount_factor=0.99,
                       separate_target_network=True,
                       target_network_update_freq=self.target_network_update_freq,
                       lr=self.lr)

        self.total_steps_counter = 0
        self.episode_steps_counter = 0
        self.last_sample_idx = 0
        self.episode_r = 0
        self.episode = 0
        self.state = 'EPISODE RUNNING'

    def update(self, s, a, r, s_prime):
        """Record an experience tuple from the environment, of the form
        (s, a, r, s')
        """

        """
        D_new_s, D_new_a, D_new_r, D_new_s_prime \
            = run_episode(env, policy, epsilon=epsilon)
        """
        print('Update. Episode #{}, Step: {}'.format(
              self.episode, self.episode_steps_counter))

        if self.state == 'EXPERIMENT ENDED':
            return
        else:
            self.D_new_s[self.episode_steps_counter] = s
            self.D_new_a[self.episode_steps_counter] = a
            self.D_new_r[self.episode_steps_counter] = r
            self.D_new_s_prime[self.episode_steps_counter] = s_prime

            self.episode_steps_counter += 1

        if self.env.terminal() or self.episode_steps_counter \
                >= self.max_steps_per_episode - 1:
            print('Episode ended.')
            self.state = 'EPISODE ENDED'
            self.next_episode()

    def next_episode(self):
        # Record the episode history
        self.episode_r_history[self.episode] = self.D_new_r.sum()

        n_new = self.D_new_s.shape[0]
        last = self.last_sample_idx
        self.D_s[last:last + n_new] = self.D_new_s
        self.D_a[last:last + n_new] = self.D_new_a
        self.D_r[last:last + n_new] = self.D_new_r
        self.D_s_prime[last:last + n_new] = self.D_new_s_prime
        self.last_sample_idx += n_new

        # Save the history of the episodes so far to disk
        print('Saving logs to disk...')
        np.save('episode_r_history.npy', self.episode_r_history)
        np.save('D_s.npy', self.D_s)
        np.save('D_a.npy', self.D_a)
        np.save('D_r.npy', self.D_r)
        np.save('D_s_prime.npy', self.D_s_prime)
        np.save('loss_history.npy', self.nfq._loss_history)
        np.save('q_predicted.npy', self.nfq._q_predicted)
        self.nfq.Q.save_weights('Q.npy', overwrite=True)
        print('Done.')

        # Run NFQ to update the Q-network
        self.nfq.fit_vectorized(self.D_s[0:self.last_sample_idx],
                                self.D_a[0:self.last_sample_idx],
                                self.D_r[0:self.last_sample_idx],
                                self.D_s_prime[0:self.last_sample_idx],
                                num_iters=1,
                                shuffle=True,
                                nb_samples=self.nb_samples,
                                sliding_window=self.sliding_window,
                                full_batch_sgd=False)  # Try True?

        if self.episode < self.num_episodes - 1:  # 0-based indexing
            self.episode += 1
            print('Simulating episode #{}.'.format(self.episode))
            self.env.reset()
            self.episode_steps_counter = 0
            self.episode_r = 0

            # Anneal the epsilon greedy exploration rate
            #"""
            if self.episode < self.num_episodes * 0.10:
                self.epsilon = 1.0
            elif self.episode < self.num_episodes * 0.20:
                self.epsilon = 0.8
            elif self.episode < self.num_episodes * 0.30:
                self.epsilon = 0.6
            elif self.episode < self.num_episodes * 0.40:
                self.epsilon = 0.4
            elif self.episode < self.num_episodes * 0.50:
                self.epsilon = 0.2
            elif self.episode < self.num_episodes * 0.60:
                self.epsilon = 0.1
            elif self.episode < self.num_episodes * 0.80:
                self.epsilon = 0.05
            else:
                self.epsilon = 0

            # TODO: Support annealing of the learning rate as well

            self.state = 'EPISODE RUNNING'
        else:
            print('Experiment complete.')
            self.state = 'EXPERIMENT ENDED'

    def next_experiment(self):
        # Initialize the experiment level variables and history logs
        self._init()


class TabularQLearningExperiment:
    def __init__(self,
                 env,
                 nb_episodes,
                 max_steps_per_episode):
        """Instantiate a tabular Q-Learning experiment.

        Parameters
        ----------
        environment : should expose the following methods and attributes:
            - state_dim
            - nb_actions
            - state
            - terminal()
            - act(action)
        """
        # ---------------------------- Initialization ----------------------------
        # Allocate NumPy arrays
        self.state = 'INITIALIZING'
        self.env = env
        self.num_episodes = nb_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.history_size = nb_episodes * max_steps_per_episode

        self._init()

    def _init(self):
        print('Initializing experiment.')
        self.D_s = np.zeros(self.history_size, dtype=np.int32)
        self.D_a = np.zeros(self.history_size, dtype=np.int32)
        self.D_r = np.zeros(self.history_size, dtype=np.float32)
        self.D_s_prime = np.zeros(self.history_size, dtype=np.int32)

        self.V_history = np.zeros((self.history_size, self.env.state_dim))
        self.episode_V_history = np.zeros((self.num_episodes, self.env.state_dim))
        self.r_history = np.zeros((self.history_size))
        self.episode_r_history = np.zeros((self.num_episodes))

        self.qlearning = TabularQLearning(state_dim=self.env.state_dim,
                                          nb_actions=self.env.nb_actions,
                                          learning_rate=0.01,  # was 0.001
                                          discount_factor=0.99,
                                          verbose=False)

        self.V_history[0, :] = \
            np.array([self.qlearning.V(s) for s in np.arange(self.env.state_dim)])
        self.total_steps_counter = 0
        self.episode_steps_counter = 0
        self.episode_r = 0
        self.episode = 0
        self.state = 'EPISODE RUNNING'

    def update(self, s, a, r, s_prime):
        """Record an experience tuple from the environment, of the form
        (s, a, r, s')
        """
        # Update the internal logs based on the updated state
        # Check if we reached a terminal state
        # Update the steps counter

        print('Update. Episode #{}, Step: {}, State: {}'.format(
              self.episode, self.episode_steps_counter, self.state))

        if self.state == 'EXPERIMENT ENDED':
            return
        else:
            self.qlearning.update(s, a, r, s_prime)

            self.V_history[self.total_steps_counter, :] = \
                np.array([self.qlearning.V(s) for s in np.arange(self.env.state_dim)])
            self.r_history[self.total_steps_counter] = r
            self.episode_r += r

            self.episode_steps_counter += 1
            self.total_steps_counter += 1

        if self.env.terminal() or self.episode_steps_counter \
                >= self.max_steps_per_episode:
            print('Episode ended.')
            self.state = 'EPISODE ENDED'
            self.next_episode()

    def next_episode(self):
        if self.episode < self.num_episodes - 1:  # 0-based indexing
            # Record the episode history
            self.episode_V_history[self.episode, :] = \
                np.array([self.qlearning.V(s) for s in np.arange(self.env.state_dim)])
            self.episode_r_history[self.episode] = self.episode_r

            self.episode += 1
            print('Simulating episode #{}.'.format(self.episode))
            self.env.reset()
            self.episode_steps_counter = 0
            self.episode_r = 0

            # Anneal the epsilon greedy exploration rate
            if self.episode < self.num_episodes * 0.80:
                self.qlearning.epsilon = 0.10
            else:
                self.qlearning.epsilon = 0

            self.state = 'EPISODE RUNNING'
        else:
            print('Experiment complete.')
            self.state = 'EXPERIMENT ENDED'

    def next_experiment(self):
        # Initialize the experiment level variables and history logs
        self._init()


if __name__ == '__main__':
    NB_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 50

    env = GridWorld()
    exp = TabularQLearningExperiment(env=env,
                                     nb_episodes=NB_EPISODES,
                                     max_steps_per_episode=MAX_STEPS_PER_EPISODE)

    for i in range(NB_EPISODES * MAX_STEPS_PER_EPISODE):
        if exp.state != 'EXPERIMENT ENDED':
            s = env.state
            a = exp.qlearning.policy(s)
            r = env.act(a)
            s_prime = env.state

            exp.update(s, a, r, s_prime)


    # ----------------------------- Visualization -----------------------------

    # Plot the value function over time
    from matplotlib import pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    states_to_plot = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]

    for s in states_to_plot:
        if s % 2:
            l = '--'
        else:
            l = '-'
        ax.plot(exp.episode_V_history[0:exp.num_episodes, s], linestyle=l)

    ax.set_xlabel('Episode')
    ax.set_ylabel('V(s)')
    ax.set_ylim([exp.episode_V_history.min() - 0.2,
                 exp.episode_V_history.max() + 0.2])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    lgd = ax.legend(states_to_plot, loc=9, bbox_to_anchor=(1.15, 1), ncol=1)
    fig.savefig('experiments/tabular_q_learning_robot/' +
                'V_episodes_{}_lr_{}_discount_{}.png'.format(
                exp.num_episodes,
                exp.qlearning.learning_rate,
                exp.qlearning.discount_factor))

    # Plot reward history
    smoothed_reward_history = \
        exp.episode_r_history[0:exp.num_episodes].reshape(-1, 10).mean(axis=1)
    x = np.arange(0, exp.num_episodes, exp.num_episodes / smoothed_reward_history.shape[0])
    fig_r = plt.figure()
    ax_r = fig_r.add_subplot(111)
    ax_r.plot(x, smoothed_reward_history)
    ax_r.set_xlim(-1, x.max());
    ax_r.set_xlabel('Episode')
    ax_r.set_ylabel('Total Reward (smoothed)')
    ax_r.grid()
    fig_r.savefig('experiments/tabular_q_learning_robot/' +
                'r_episodes_{}_lr_{}_discount_{}.png'.format(
                exp.num_episodes,
                exp.qlearning.learning_rate,
                exp.qlearning.discount_factor))
