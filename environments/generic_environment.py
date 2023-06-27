import numpy as np
import util

random = np.random.random
randint = np.random.randint


class GenericEnvironment:

    # depth of each dimension
    def __init__(self, dimensions, actions, start_state, number_of_possible_states, actions_str=None):
        # depth of each dimension
        self.dimensions = dimensions
        # actions
        self.actions = actions
        # starting state
        self.start_state = start_state
        self._actions_str = list()
        if actions_str is None:
            for action in actions:
                self._actions_str.append(str(action))
        else:
            self._actions_str = actions_str
        self.number_of_possible_states = number_of_possible_states

    def state_to_int(self, state):
        return util.binary_to_decimal(state)

    def int_state_to_tuple(self, int_state):
        return util.decimal_to_binary(int_state, len(self.dimensions))

    def action_to_str(self, action):
        return self.int_action_to_str(self.action_to_int(action))

    def int_action_to_str(self, int_action):
        return self._actions_str[int_action]

    # since action is a tuple in this example but not in the QTable
    def action_to_int(self, action):
        return self.actions.index(action)

    def get_reward(self, state, action, next_state):
        raise Exception("get_reward was not properly implemented")

    def get_next_state(self, state, action):
        raise Exception("get_next_state was not properly implemented")

    def get_possible_qualities_and_actions(self, q_table, state):
        raise Exception("get_possible_qualities_and_actions was not properly implemented")

    def done(self, state):
        raise Exception("done was not properly implemented")

    def get_possible_actions(self, state):
        raise Exception("get_possible_actions was not properly implemented")

    def action_possible(self, state, action):
        if state[self.action_to_int(action)] == 0:
            return True
        return False

    def get_start_state(self):
        return list(self.start_state)
