'''Implementation of the Regularized Convolutional Neural Fitted Q-Iteration
(RC-NFQ) algorithm.
'''

import numpy as np
import time
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop
from keras.callbacks import RemoteMonitor
from keras.callbacks import ModelCheckpoint


class NFQ:
    """Regularized Convolutional Neural Fitted Q-Iteration (RC-NFQ)

    References: 
      - Riedmiller, Martin. "Neural fitted Q iteration-first experiences
        with a data efficient neural reinforcement learning method." Machine
        Learning: ECML 2005. Springer Berlin Heidelberg, 2005. 317-328.

      - Mnih, Volodymyr, et al. "Human-level control through deep
        reinforcement learning." Nature 518.7540 (2015): 529-533.

      - Lin, Long-Ji. "Self-improving reactive agents based on reinforcement
        learning, planning and teaching." Machine learning 8.3-4 (1992):
        293-321.

      - Harrigan, Cosmo (2016). "Regularized Convolutional Neural Fitted
        Q-Iteration." Manuscript in preparation.
    """

    def __init__(self,
                 state_dim,
                 nb_actions,
                 terminal_states,
                 convolutional=False,
                 mlp_layers=[20, 20],
                 discount_factor=0.99,
                 separate_target_network=False,
                 target_network_update_freq=None,
                 lr=0.01,
                 max_iters=20000):
        """Create an instance of the NFQ algorithm for a particular agent and
        environment.

        Parameters
        ----------
        state_dim : The state dimensionality. An integer if 
                    convolutional = False, a 2D tuple otherwise.
        nb_actions : The number of possible actions
        terminal_states : The integer indices of the terminal states
        convolutional : Boolean. When True, uses convolutional neural networks 
                        and dropout regularization. Otherwise, uses a simple
                        MLP.
        mlp_layers : A list consisting of an integer number of neurons for each
                     hidden layer. Default = [20, 20]. For convolutional = 
                     False.
        discount_factor : The discount factor for Q-learning.
        separate_target_network: boolean - If True, then it will use a separate
                                 Q-network for computing the targets for the
                                 Q-learning updates, and the target network
                                 will be updated with the parameters of the
                                 main Q-network every
                                 target_network_update_freq iterations.
        target_network_update_freq : The frequency at which to update the 
                                     target network.
        lr : The learning rate for the RMSprop gradient descent algorithm.
        max_iters : The maximum number of iterations that will be performed.
                    Used to allocate memory for NumPy arrays. Default = 20000.
        max_q_predicted : The maximum number of Q-values that will be predicted.
                          Used to allocate memory for NumPy arrays. Default = 
                          100000.
        """
        self.convolutional = convolutional
        self.separate_target_network = separate_target_network
        self.k = 0  # Keep track of the number of iterations
        self.discount_factor = discount_factor
        self.nb_actions = nb_actions
        self.state_dim = state_dim
        self.lr = lr

        self._loss_history = np.zeros((max_iters))
        self._loss_history_test = np.zeros((max_iters))

        self._q_predicted = np.empty((max_q_predicted), dtype=np.float32)
        self._q_predicted[:] = np.NAN
        self._q_predicted_counter = 0

        self.terminal_states = terminal_states

        if self.convolutional:
            self.Q = self._init_convolutional_NFQ()
        else:
            self.Q = self._init_MLP(mlp_layers=mlp_layers)

        if self.separate_target_network:
            assert target_network_update_freq is not None

            if self.convolutional:
                self.Q_target = self._init_convolutional_NFQ()
            else:
                self.Q_target = self._init_MLP(mlp_layers=mlp_layers)
            # Copy the initial weights from the Q network
            self.Q_target.set_weights(self.Q.get_weights())

            self.target_network_update_freq = target_network_update_freq

    def __str__(self):
        """Print the current Q function and value function."""
        string = ""
        if self.convolutional:
            string += 'Tabular values not available for NFQ with a ' + \
                      'Convolutional Neural Network function approximator.'
        else:
            for s in np.arange(self.state_dim):
                for a in np.arange(self.nb_actions):
                    r = self._Q_value(s, a)
                    string += 'Q(s={}, a={}) = {}\n'.format(s, a, r)

            for s in np.arange(self.state_dim):
                v = self._greedy_action_value(s)
                string += 'V({}) = {}\n'.format(s, v)

        return string

    def fit_vectorized(self, D_s, D_a, D_r, D_s_prime,
                       num_iters=1,
                       shuffle=False,
                       nb_samples=None,
                       sliding_window=None,
                       full_batch_sgd=False,
                       validation=True):
        """Run an iteration of the RC-NFQ algorithm and update the Q function.
        The implementation is vectorized for improved performance.

        The function requires a set of interactions with the environment. 
        They consist of experience tuples of the form (s, a, r, s_prime),
        stored in 4 parallel arrays.

        Parameters
        ----------
        D_s : A list of states s for each experience tuple
        D_a: A list of actions a for each experience tuple
        D_r : A list of rewards r for each experience tuple
        D_s_prime : A list of states s_prime for each experience tuple
        num_iters : The number of epochs to run per batch. Default = 1.
        shuffle : Whether to shuffle the data before training. Default = False.
        nb_samples : If specified, uses nb_samples samples from the experience
                     tuples selected without replacement. Otherwise, all eligible
                     samples are used.
        sliding_window : If specified, only the last nb_samples samples will be
                         eligible for use. Otherwise, all samples are eligible.
        full_batch_sgd : Boolean. Determines whether RMSprop will use 
                         full-batch or mini-batch updating. Default = False.
        validation : Boolean. If True, a validation set will be used consisting
                     of the last 10% of the experience tuples, and the validation 
                     loss will be monitored. Default = True.
        """
        if validation:
            # Split the data into 90% training / 10% validation sets
            n = int(0.90 * D_s.shape[0])

            D_s_train = D_s[0:n]
            D_a_train = D_a[0:n]
            D_r_train = D_r[0:n]
            D_s_prime_train = D_s_prime[0:n]

            D_s_test = D_s[n:]
            D_a_test = D_a[n:]
            D_r_test = D_r[n:]
            D_s_prime_test = D_s_prime[n:]

        else:
            D_s_train, D_a_train, D_r_train, D_s_prime_train = D_s, D_a, D_r, D_s_prime

        if sliding_window is not None:
            if sliding_window < D_s_train.shape[0]:
                D_s_train = D_s_train[-sliding_window:]
                D_a_train = D_a_train[-sliding_window:]
                D_r_train = D_r_train[-sliding_window:]
                D_s_prime_train = D_s_prime_train[-sliding_window:]

        if shuffle:
            indices = np.arange(D_s_train.shape[0])
            np.random.shuffle(indices)
            D_s_train = D_s_train[indices]
            D_a_train = D_a_train[indices]
            D_r_train = D_r_train[indices]
            D_s_prime_train = D_s_prime_train[indices]

        if nb_samples is not None:
            nb_samples = min(nb_samples, D_s_train.shape[0])
            indices = np.random.choice(D_s_train.shape[0], nb_samples)
            D_s_train = D_s_train[indices]
            D_a_train = D_a_train[indices]
            D_r_train = D_r_train[indices]
            D_s_prime_train = D_s_prime_train[indices]

        print('k: {}, update frequency: {}'.format(self.k, self.target_network_update_freq))

        if self.separate_target_network:
            # Update the target Q-network every target_network_update_freq
            # iterations with the parameters from the main Q-network
            if self.k % self.target_network_update_freq == 0:
                print('* Updating target Q-network parameters.')
                self.Q_target.set_weights(self.Q.get_weights())

        # P contains the pattern set of inputs and targets
        P_input_values_train, P_target_values_train \
            = self._generate_pattern_set_vectorized(D_s_train, D_a_train, D_r_train, D_s_prime_train)

        P_input_values_test, P_target_values_test \
            = self._generate_pattern_set_vectorized(D_s_test, D_a_test, D_r_test, D_s_prime_test)

        if self.convolutional:
            P_input_values_states_train = P_input_values_train[0]
            P_input_values_actions_train = P_input_values_train[1]
            P_input_values_states_test = P_input_values_test[0]
            P_input_values_actions_test = P_input_values_test[1]

            checkpointer = ModelCheckpoint(filepath="/tmp/nfq_weights.{}.hdf5".format(self.k),
                                           verbose=1,
                                           save_best_only=False)

            if full_batch_sgd:
                if validation:
                    hist = self.Q.fit({'input_state': P_input_values_states_train,
                                       'input_action': P_input_values_actions_train,
                                       'output_q_value': P_target_values_train},
                                       nb_epoch=num_iters,
                                       batch_size=P_target_values_train.shape[0],
                                       validation_data=\
                                       {'input_state': P_input_values_states_test,
                                       'input_action': P_input_values_actions_test,
                                       'output_q_value': P_target_values_test},
                                       callbacks=[checkpointer])
                else:
                    hist = self.Q.fit({'input_state': P_input_values_states_train,
                                       'input_action': P_input_values_actions_train,
                                       'output_q_value': P_target_values_train},
                                       nb_epoch=num_iters,
                                       batch_size=P_target_values.shape[0],
                                       callbacks=[checkpointer])
            else:
                if validation:
                    hist = self.Q.fit({'input_state': P_input_values_states_train,
                                       'input_action': P_input_values_actions_train,
                                       'output_q_value': P_target_values_train},
                                       nb_epoch=num_iters,
                                       validation_data=\
                                       {'input_state': P_input_values_states_test,
                                       'input_action': P_input_values_actions_test,
                                       'output_q_value': P_target_values_test},
                                       callbacks=[checkpointer])
                else:
                    hist = self.Q.fit({'input_state': P_input_values_states_train,
                                       'input_action': P_input_values_actions_train,
                                       'output_q_value': P_target_values_train},
                                       nb_epoch=num_iters,
                                       callbacks=[checkpointer])
        else:
            if full_batch_sgd:
                if validation:
                    hist = self.Q.fit(P_input_values_train,
                                      P_target_values_train,
                                      nb_epoch=num_iters,
                                      batch_size=P_target_values_train.shape[0],
                                      validationI_data=(P_input_values_test,
                                                        P_target_values_test),
                                      callbacks=[checkpointer])
            else:
                hist = self.Q.fit(P_input_values,
                                  P_target_values,
                                  nb_epoch=num_iters,
                                  validationI_data=(P_input_values_test,
                                                    P_target_values_test),
                                  callbacks=[checkpointer])

        self._loss_history[self.k] = hist.history['loss'][0]

        self._last_loss_history = hist.history['loss']

        if validation:
            self._loss_history_test[self.k] = hist.history['val_loss'][0]
            self._last_loss_history_test = hist.history['val_loss']

        self.k += num_iters

    def greedy_action(self, s):
        """Return the action that maximizes expected reward from a given state.
           TODO: Vectorize this function for improved performance.
        """
        Q_value = np.zeros(self.nb_actions)
        for a in np.arange(self.nb_actions):
            Q_value[a] = self._Q_value(s, a)

            print('Q-value of action {}: {}'.format(a, Q_value[a]))

        greedy_action = np.argmax(Q_value)

        self._q_predicted[self._q_predicted_counter] = Q_value[greedy_action]
        print('Stored predicted Q-value of {} for action {}'.\
              format(self._q_predicted[self._q_predicted_counter], greedy_action))
        self._q_predicted_counter += 1

        return greedy_action

    def value_function(self):
        values = np.zeros((self.state_dim))
        for s in np.arange(self.state_dim):
            values[s] = self._greedy_action_value(s)
        return values

    def _init_MLP(self, mlp_layers):
        """Initialize the MLP that corresponds to the Q function.

        Parameters
        ----------
        state_dim : The state dimensionality
        nb_actions : The number of possible actions
        mlp_layers : A list consisting of an integer number of neurons for each
                     hidden layer. Default = [20, 20]
        """
        model = Sequential()
        for i in range(len(mlp_layers)):
            if i == 0:
                model.add(Dense(mlp_layers[i],
                          input_dim=self.state_dim + self.nb_actions))
            else:
                model.add(Dense(mlp_layers[i]))
            model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('relu'))
        rmsprop = RMSprop()
        model.compile(loss='mean_squared_error', optimizer=rmsprop)
        return model

    def _init_convolutional_NFQ(self):
        """Initialize a convolutional NFQ network.

        TODO: Allow customization of the network topology.
        """
        # ConvNet to process the input image
        cnn = Sequential()

        cnn.add(Convolution2D(nb_filter=16, nb_row=8, nb_col=8,
                              input_shape=(1, 64, 64), subsample=(4, 4)))
        cnn.add(Activation('relu'))

        cnn.add(Dropout(0.25))
        cnn.add(Convolution2D(nb_filter=32, nb_row=4, nb_col=4,
                              subsample=(2, 2)))
        cnn.add(Activation('relu'))

        cnn.add(Dropout(0.25))
        cnn.add(Flatten())

        # Combine the ConvNet output with the action to get the Q-value estimate
        graph = Graph()

        # State and action inputs
        graph.add_input(name='input_state', input_shape=self.state_dim)
        graph.add_input(name='input_action', input_shape=(self.nb_actions,))
        graph.add_node(cnn, name='cnn', input='input_state')

        graph.add_node(Dense(256),
                       name='nfq_dense_0',
                       inputs=['cnn', 'input_action'])
        graph.add_node(Activation('relu'),
                       name='nfq_activation_0',
                       input='nfq_dense_0')

        graph.add_node(Dropout(0.25), name='nfq_dropout', input='nfq_dense_0')

        graph.add_node(Dense(1),
                       name='nfq_output_dense',
                       input='nfq_dropout')

        # Unscaled output
        graph.add_output(name='output_q_value', input='nfq_output_dense')

        graph.compile(loss={'output_q_value': 'mse'}, optimizer=RMSprop(lr=self.lr))
        return graph

    def _generate_pattern_set_vectorized(self, D_s, D_a, D_r, D_s_prime):
        """Generate pattern set. Vectorized version for improved performance.

        A pattern set consists of a set of input and target tuples, where
        the inputs contain states and actions that occurred in the
        environment, and the targets are calculated based on the Bellman
        equation using the reward from the environment and the Q-value
        estimate for the successor state using the current Q function.

        Parameters
        ----------
        D_s : A list of states s for each experience tuple
        D_a: A list of actions a for each experience tuple
        D_r : A list of rewards r for each experience tuple
        D_s_prime : A list of states s_prime for each experience tuple
        """
        # Perform a forward pass through the Q-value network as a batch with
        # all the samples from D at the same time for efficiency

        # P contains the pattern set of inputs and targets
        if self.convolutional:
            P_input_values_actions = \
                self._one_hot_encode_actions_vectorized(D_a)
            P_input_values = D_s, P_input_values_actions
        else:
            P_input_values = \
                self._one_hot_encode_states_actions_vectorized(D_s, D_a)

        if self.separate_target_network:
            target_network = self.Q_target
        else:
            target_network = self.Q

        P_target_values = D_r + self.discount_factor * \
            self._greedy_action_value_vectorized(s=D_s_prime,
                                                 Q_network=target_network)

        return P_input_values, P_target_values

    def _Q_value(self, s, a):
        """Calculate the Q-value of a state, action pair
        """
        if self.convolutional:
            a = np.array((a)).reshape(1, 1)
            s = s.reshape(1, 1, s.shape[0], s.shape[1])
            return self._Q_value_vectorized(s, a, self.Q)
        else:
            X = self._encode_input(s, a)
            # Perform a forward pass of the Q-network
            output = self.Q.predict(X)[0][0]
        return output

    def _Q_value_vectorized(self, s, a, Q_network):
        """Calculates the Q-values of two vectors of state, action pairs
        """
        if self.convolutional:
            a_one_hot = self._one_hot_encode_actions_vectorized(a)
            output = Q_network.predict({'input_state': s,
                                        'input_action': a_one_hot})
            output = output['output_q_value'].reshape(a.shape[0])
        else:
            X = self._one_hot_encode_states_actions_vectorized(s, a)
            # Perform a batch forward pass of the Q-network
            output = Q_network.predict(X).reshape(X.shape[0])

        # Set the Q-value of terminal states to zero
        if self.terminal_states is not None:
            for terminal_state in self.terminal_states:
                output[output == terminal_state] = 0

        return output

    def _greedy_action_value_vectorized(self, s, Q_network):
        """Calculate the value of each state in a state vector assuming the
        action with the highest Q-value is performed
        """
        nb_states = s.shape[0]
        # Construct an array of shape (nb_states, nb_actions) to store the
        # Q(s, a) estimates for each state, action pair. The action value is
        # stored in a parallel array to the state array.
        action_value = np.zeros((nb_states, self.nb_actions))

        # Run a batch forward pass through the Q-network to calculate the
        # estimated action value for a given action across all the states
        # in the state vector
        for a in np.arange(self.nb_actions):
            action_vector = np.empty((nb_states), dtype=np.int64)
            action_vector.fill(a)
            action_value[:, a] = \
                self._Q_value_vectorized(s, action_vector, Q_network=Q_network)

        greedy_action_value = np.max(action_value, axis=1)

        return greedy_action_value

    def _one_hot_encode_states_actions_vectorized(self, states, actions):
        """Encode a matrix of (state, action) pairs into one-hot vector
        representations
        """
        n_dim = states.shape[0]
        encoding_length = self.state_dim + self.nb_actions
        M = np.zeros((n_dim, encoding_length))
        # Mark the selected states as 1
        M[np.arange(n_dim), 0 + states] = 1
        # Mark the selected actions as 1. To calculate the action indices,
        # they need to be added to the end of the state portion of the vector.
        M[np.arange(n_dim), self.state_dim + actions] = 1
        return M

    def _one_hot_encode_actions_vectorized(self, actions):
        """Encode a vector of actions into one-hot vector representations
        """
        n_dim = actions.shape[0]
        encoding_length = self.nb_actions
        M = np.zeros((n_dim, encoding_length))
        # Mark the selected actions as 1
        M[np.arange(n_dim), 0 + actions] = 1
        return M

    def _encode_input(self, s, a):
        """Encode a (state, action) pair into a one-hot vector representation
        """
        # Encode the state as a one-hot vector
        state_one_hot = self._one_hot_encode(cls=s, nb_classes=self.state_dim)
        # Encode the action as a one-hot vector
        action_one_hot = self._one_hot_encode(cls=a, nb_classes=self.nb_actions)
        # Concatenate the state and action vectors
        X = np.array(state_one_hot + action_one_hot)
        X = X.reshape(1, self.state_dim + self.nb_actions)
        return X

    def _one_hot_encode(self, cls, nb_classes):
        """Convert an integer into one-hot vector representation
        """
        one_hot = [0] * nb_classes
        one_hot[cls] = 1
        return one_hot
