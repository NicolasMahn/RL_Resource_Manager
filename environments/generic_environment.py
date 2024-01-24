import numpy as np
import tensorflow as tf

from resources import util

# Random number generators
random = np.random.random
randint = np.random.randint
randint_with_weighted_sqrt = util.weighted_randint


class GenericEnvironment:

    def __init__(self, dimensions, actions, start_state, number_of_possible_states=None, actions_str=None):
        # Initialization of the environment
        self.dimensions = dimensions  # Depth of each dimension in the state space
        self.actions = actions  # Available actions in the environment
        self.result = start_state  # Starting state of the environment
        self._actions_str = [str(action) for action in actions] if actions_str is None else actions_str
        self.number_of_possible_states = number_of_possible_states  # Total number of possible states

    @staticmethod
    def state_to_int(state):
        # Converts a state represented in binary to its decimal equivalent
        return util.binary_to_decimal(state)

    def int_state_to_tuple(self, int_state):
        # Converts an integer state back to its binary (tuple) representation
        return util.decimal_to_binary(int_state, len(self.dimensions))

    def action_to_str(self, action):
        # Converts an action to its string representation
        return self.int_action_to_str(self.action_to_int(action))

    def int_action_to_str(self, int_action):
        # Converts an integer action to its string representation
        return self._actions_str[int_action]

    def action_to_int(self, action):
        # Converts an action (tuple) to its index in the action list
        return self.actions.index(action)

    def action_possible(self, state, action):
        # Check if a given action is possible in the current state
        if state[self.action_to_int(action)] == 0:
            return True
        return False

    def get_start_state(self):
        # Returns the starting state of the environment
        return list(self.result)

    # Abstract methods that should be implemented in subclasses
    def get_reward(self, state, action, next_state):
        # Abstract method to calculate the reward given a state, action, and next state
        raise Exception("get_reward was not properly implemented")

    def get_next_state(self, state, action):
        # Abstract method to determine the next state given a state and action
        raise Exception("get_next_state was not properly implemented")

    def get_possible_qualities_and_actions(self, q_table, state):
        # Abstract method to determine possible qualities and actions in a given state
        raise Exception("get_possible_qualities_and_actions was not properly implemented")

    def done(self, state):
        # Abstract method to check if the current state is a terminal state
        raise Exception("done was not properly implemented")

    def get_possible_actions(self, state):
        # Abstract method to get possible actions in the current state
        raise Exception("get_possible_actions was not properly implemented")

    def check_if_step_correct(self, state, action, next_state):
        # Abstract method to calculate the true or actual reward given a state, action, and next state
        raise Exception("get_true_reward was not properly implemented "
                        "or it might not be possible with this environemnt.")

    def to_tensor_state(self, state):
        # Function to prepare the state for Deep Q-Network (DQN) processing
        return tf.convert_to_tensor(self.pad_state(state))

    def pad_state(self, state):
        # Abstract method to pad state to the maximum possible number of tasks
        raise Exception("pad_state was not properly implemented "
                        "or it might not be possible with this environemnt.")
