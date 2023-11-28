import numpy as np
import tensorflow as tf

from .generic_environment import GenericEnvironment
from resources import util

# Random number generators
random_choice = np.random.choice
random = np.random.random
randint = np.random.randint
weighted_randint = util.weighted_randint


class Jm_tf_T_JSSProblem(GenericEnvironment):

    def __init__(self, max_numb_of_tasks, test_set, fixed_max_numbers,
                 high_numb_of_tasks_preference):
        # A state needs the following information for all n*tasks in it:
        # # id
        # # child_foreign_key
        # # nonpreemtive_flag
        # # lead_time
        # # processing_time_total
        # # processing_time_todo
        # # deadline (for simplicity only in int)
        # # done_flag

        # An action is simply the id of a task
        self.numb_of_attributes = 9

        # Initialize TimeManagement environment with specific parameters
        self.fixed_max_numbers = fixed_max_numbers  # Flag to keep number of tasks constant
        self.high_numb_of_tasks_preference = high_numb_of_tasks_preference  # Preference for high number of tasks
        self.max_numb_of_tasks = max_numb_of_tasks  # Maximum number of tasks in the environment

        # Handling test sets if provided
        self.test_set = test_set
        self.test_set_tasks = [item["tasks"] for item in test_set] if test_set is not None else None

        # Defining dimensions for the environment
        dimensions = [self.numb_of_attributes, self.max_numb_of_tasks]

        self.numb_of_tasks = self.max_numb_of_tasks

        # Initialize the state elements
        self.ids = 0
        self.child_foreign_keys = 1
        self.nonpreemtive_flag = 2
        self.lead_time_total = 3
        self.lead_time_todo = 4
        self.processing_time_total = 5
        self.processing_time_todo = 6
        self.deadline = 7
        self.done_flag = 8

        # Generate possible actions
        actions = np.arange(0, self.numb_of_tasks).tolist()

        self.history = list()

        # Call to superclass constructor
        super().__init__(dimensions=dimensions, actions=actions, start_state=np.zeros(max_numb_of_tasks))

    def get_specific_state(self, ids, child_foreign_keys, nonpreemtive_flag, lead_time_total, lead_time_todo,
                           processing_time_total, processing_time_todo, deadline, done_flag):
        # Function to get a specific state based on provided attributes
        return list([ids, child_foreign_keys, nonpreemtive_flag, lead_time_total, lead_time_todo, processing_time_total,
                     processing_time_todo, deadline, done_flag])

    def get_specific_state_list(self, list_):
        # Function to get a specific state based on provided attributes
        return list([list_[self.ids], list_[self.child_foreign_keys], list_[self.nonpreemtive_flag],
                     list_[self.lead_time_total], list_[self.lead_time_todo], list_[self.processing_time_total],
                     list_[self.processing_time_todo], list_[self.deadline],
                     list_[self.done_flag]])

    def get_start_state(self):
        # Function to initialize the starting state of the environment
        ids = np.arange(0, self.numb_of_tasks)
        child_foreign_keys = util.validate_child_elements(random_choice(np.arange(-1, self.numb_of_tasks),
                                                                        size=self.numb_of_tasks,
                                                                        p=[1 / 3] +
                                                                          [2 / (3 * self.numb_of_tasks)] *
                                                                          self.numb_of_tasks))
        nonpreemtive_flag = random_choice(np.arange(0, 2), size=self.numb_of_tasks)
        lead_time_total = random_choice(np.arange(0, 5),  size=self.numb_of_tasks, p=[1 / 2] + [1 / 8] * 4)
        lead_time_todo = lead_time_total
        processing_time_total = random_choice(np.arange(1, 10), size=self.numb_of_tasks)
        processing_time_todo = processing_time_total
        deadline = random_choice(np.arange(10, 50), size=self.numb_of_tasks)
        done_flag = np.zeros(self.numb_of_tasks, dtype=int)

        return list([ids, child_foreign_keys, nonpreemtive_flag, lead_time_total, lead_time_todo, processing_time_total,
                     processing_time_todo, deadline, done_flag])

    def done(self, state):
        # Function to check if the current state is a terminal state
        if sum(state[self.done_flag]) == self.numb_of_tasks:
            print("done")
            return True
        else:
            return False

    def get_reward(self, state, action, next_state):
        # Function to calculate the reward based on the state, action, and next state
        reward = 0.1  # small reward for each step
        j = 0

        #for i in range(0, self.numb_of_tasks):
        #    if next_state[self.deadline][i] <= 0 and next_state[self.done_flag][i] != 1:
        #        reward -= 100 * (next_state[self.deadline][i]*-1+1)
        #    j += next_state[self.deadline][i]

        reward += np.average(j)

        return reward

    def get_next_state(self, state, action):
        # Function to determine the next state based on the current state and action
        next_state = list()
        for s in state:
            next_state.append(list(s))

        if next_state[self.nonpreemtive_flag][action] == 0:
            next_state[self.processing_time_todo][action] = max(0, next_state[self.processing_time_todo][action]-1)
            for i in range(0, self.numb_of_tasks):
                if next_state[self.done_flag][i] == 0 and next_state[self.processing_time_todo][i] == 0:
                    next_state[self.lead_time_todo][i] = max(0, next_state[self.lead_time_todo][i] - 1)
                if next_state[self.done_flag][i] == 0:
                    next_state[self.deadline][i] -= 1
        else:
            next_state[self.processing_time_todo][action] = 0
            for i in range(0, self.numb_of_tasks):
                if next_state[self.done_flag][i] == 0 and next_state[self.processing_time_todo][i] == 0:
                    next_state[self.lead_time_todo][i] = max(0, next_state[self.lead_time_todo][i] -
                                                             next_state[self.processing_time_total][action])
                if next_state[self.done_flag][i] == 0:
                    next_state[self.deadline][i] = next_state[self.deadline][i] - \
                                                   next_state[self.processing_time_total][action]
                next_state[self.processing_time_todo][action] = 0
        if next_state[self.processing_time_todo][action] == 0 and next_state[self.lead_time_todo][action] == 0:
            next_state[self.done_flag][action] = 1

        self.history.append((state, action))
        return next_state

    def get_possible_actions(self, state):
        # Function to determine possible actions in the current state
        possible_actions = []
        impossible_actions = []

        for action in self.actions:
            if (action in state[self.child_foreign_keys]) or (state[self.done_flag][action] != 0) or \
                    (state[self.lead_time_todo][action] != 0 and state[self.processing_time_todo][action] == 0):
                impossible_actions.append(action)
            else:
                possible_actions.append(action)

        return possible_actions, impossible_actions

    def state_for_dqn(self, state):
        # Function to prepare the state for Deep Q-Network (DQN) processing
        state_padded = list()
        for s in state:
            s_padded = np.pad(s, (0, self.max_numb_of_tasks - len(s)), constant_values=-1)
            state_padded.append(s_padded)
        return tf.convert_to_tensor(state_padded)

    def get_result(self):
        # Function to retrieve the history
        return self.history
