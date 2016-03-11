'''
Environment definitions for the RC-NFQ algorithm.
'''

import numpy as np
from scipy import stats


class SimpleWorld:
    """Basic test world.
    """
    def __init__(self, max_steps):
        self.actions = [0, 1]
        self.nb_actions = len(self.actions)
        self.states = [0, 1, 2, 3]
        self.nb_states = len(self.states)
        self.terminal_states = [0, 0, 0, 1]
        self.start_state = 1
        self.state = self.start_state
        self.rewards = [-1, -0.25, -0.25, 1]
        self.max_steps = max_steps

        stochastic = True
        if stochastic:
            self.T = {
                0: np.array(((0.9, 0.1, 0, 0),
                             (0.9, 0, 0.1, 0),
                             (0, 0.9, 0.1, 0),
                             (0, 0, 0, 1))),
                1: np.array(((0.1, 0.9, 0, 0),
                             (0.1, 0, 0.9, 0),
                             (0, 0.1, 0, 0.9),
                             (0, 0, 0, 1)))
            }
        else:
            self.T = {
                0: np.array(((1, 0, 0, 0),
                             (1, 0, 0, 0),
                             (0, 1, 0, 0),
                             (0, 0, 0, 1))),
                1: np.array(((0, 1, 0, 0),
                             (0, 0, 1, 0),
                             (0, 0, 0, 1),
                             (0, 0, 0, 1)))
            }

    def reset(self):
        self.state = self.start_state

    def terminal(self):
        return self.terminal_states[self.state]

    def act(self, action):
        try:
            assert not self.terminal()
        except AssertionError as e:
            e.args += ('Further action not permitted: terminal state ' +
                       ' reached. Episode is over.',)
            raise

        probs = self.T[action][self.state, :]
        pmf = stats.rv_discrete(name='pmf',
                                values=(self.states, probs))
        successor_state = pmf.rvs()
        self.state = successor_state

        r = self.rewards[successor_state]
        return r


class GridWorld:
    """Defines a simple MDP environment with 3 states.
    """
    def __init__(self, max_steps=50, start_state=0):
        """Initializes the environment.
        """
        # Define the actions:
        #     0=left, 1=right, 2=up, 3=down
        self.actions = [0, 1, 2, 3]
        self.nb_actions = len(self.actions)
        self.max_steps = max_steps

        # Define the transition matrix for each action
        # Format of entries in the transition probabilities dictionary:
        #     a: probability_matrix
        # where a is the action, and probability_matrix consists of row
        # vectors of probabilities where the row index corresponds to
        # the current state and the column index corresponds to the successor
        # state.
        self.T = {}
        for a in np.arange(self.nb_actions):
            self.T[a] = np.load('gridworld_{}.npy'.format(a))

        # Format of entries in the rewards dictionary:
        #     s: r
        self.rewards = np.load('gridworld_r.npy')

        # Load the states and terminal states
        self.terminal_states = np.load('gridworld_terminal.npy')
        self.terminal_state_indices = np.where(self.terminal_states == 1)
        self.states = np.arange(self.terminal_states.shape[0])
        self.nb_states = len(self.states)
        self.state_dim = self.nb_states

        # Define the start state and set the current state to it
        self.start_state = start_state
        self.state = self.start_state

    def act(self, action):
        try:
            assert not self.terminal()
        except AssertionError as e:
            e.args += ('Further action not permitted: terminal state ' +
                       ' reached. Episode is over.',)
            raise

        probs = self.T[action][self.state, :]
        pmf = stats.rv_discrete(name='pmf',
                                values=(self.states, probs))
        successor_state = pmf.rvs()
        self.state = successor_state

        r = self.rewards[successor_state]
        return r

    def terminal(self):
        return self.terminal_states[self.state]

    def reset(self):
        self.state = self.start_state


class SimpleRobot:
    """Defines a simple robot environment
    """
    def __init__(self, max_steps=50):
        """Initializes the environment.
        """
        # Actions: stop, forward, backward, left, right
        self.actions = np.arange(5)
        self.nb_actions = len(self.actions)
        self.max_steps = 50

        # States: a discretization of sonar distances into 10 buckets
        self.states = np.arange(10)
        self.state_dim = len(self.states)

    def terminal(self):
        return False

    def reset(self):
        pass


class VisionRobot:
    """Defines a robot environment where the state input is a video camera
    stream
    """
    def __init__(self, max_steps=50):
        """Initializes the environment.
        """
        # Actions: forward, backward, left, right
        self.actions = np.arange(4)
        self.nb_actions = len(self.actions)
        self.max_steps = 50

        # States: grayscale 64x64 camera images
        self.state_dim = (1, 64, 64)

    def terminal(self):
        return False

    def reset(self):
        pass
