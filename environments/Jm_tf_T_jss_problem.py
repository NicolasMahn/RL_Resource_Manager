import numpy as np
import tensorflow as tf

from .generic_environment import GenericEnvironment
from resources import util, data_generation

# Random number generators
random_choice = np.random.choice
random = np.random.random
randint = np.random.randint
weighted_randint = util.weighted_randint


class Jm_tf_T_JSSProblem(GenericEnvironment):

    def __init__(self, max_numb_of_tasks, test_set, fixed_max_numbers,
                 high_numb_of_tasks_preference, dir_name):
        self.env_name = "[J,m=1|pmtn,nowait,tree,nj,t,f,gj=1|avg(T)]"
        self.dir_name = dir_name

        # A state needs the following information for all n*tasks in it:
        # # id
        # # child_foreign_key
        # # nonpreemtive_flag
        # # lead_time
        # # processing_time_total
        # # processing_time_todo
        # # deadline (for simplicity only in int)
        # # done_flag
        # # is task ready (i.e. has no parent)

        # An action is simply the id of a task
        self.numb_of_attributes = 10

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
        # These are the indexes of the state, enabling better code readability
        self.ids = 0
        self.child_foreign_keys = 1
        self.nonpreemtive_flag = 2
        self.lead_time_total = 3
        self.lead_time_todo = 4
        self.processing_time_total = 5
        self.processing_time_todo = 6
        self.deadline = 7
        self.done_flag = 8
        self.is_task_ready = 9

        # Generate possible actions
        actions = np.arange(0, self.numb_of_tasks).tolist()

        self.history = list()

        # Call to superclass constructor
        super().__init__(dimensions=dimensions, actions=actions, start_state=np.zeros(max_numb_of_tasks))

    def get_specific_state(self, ids, child_foreign_keys, nonpreemtive_flag, lead_time_total, lead_time_todo,
                           processing_time_total, processing_time_todo, deadline, done_flag, is_task_ready):
        # Function to get a specific state based on provided attributes
        return list([ids, child_foreign_keys, nonpreemtive_flag, lead_time_total, lead_time_todo, processing_time_total,
                     processing_time_todo, deadline, done_flag, is_task_ready])

    def get_specific_state_list(self, list_):
        # Function to get a specific state based on provided attributes
        return self.get_specific_state(list_[self.ids], list_[self.child_foreign_keys], list_[self.nonpreemtive_flag],
                                       list_[self.lead_time_total], list_[self.lead_time_todo],
                                       list_[self.processing_time_total], list_[self.processing_time_todo],
                                       list_[self.deadline], list_[self.done_flag], list_[self.is_task_ready])

    def get_start_state(self, num_episode: int):
        temp_list = data_generation.get_start_state(self.env_name, self.numb_of_tasks, num_episode, self.dir_name)
        return temp_list

    def done(self, state):
        # Function to check if the current state is a terminal state
        if sum(state[self.done_flag]) == self.numb_of_tasks:
            return True
        else:
            return False

    def check_if_step_correct(self, state, action, next_state):
        min_time = float('inf')
        chosen_action = None
        for i, processing_time in enumerate(state[self.processing_time_todo]):
            if state[self.is_task_ready][i] == 1 and state[self.done_flag][i] == 0 and processing_time < min_time:
                min_time = processing_time
                chosen_action = i

        if chosen_action != action:
            return -1
        else:
            return 1

    def get_reward(self, state, action, next_state):
        # Function to calculate the reward based on the state, action, and next state
        reward = 0.1  # small reward for each step
        j = 0

        # for i in range(0, self.numb_of_tasks):
        # if next_state[self.deadline][i] <= 0 and next_state[self.done_flag][i] != 1:
        #      reward -= 2 * (next_state[self.deadline][i] * -1 + 1)
        # if state[self.deadline][i] <= 0 and state[self.done_flag][i] != 1:
        #    reward += 10 * (next_state[self.deadline][i] * -1 + 1)
        #   j += next_state[self.deadline][i]

        # reward += np.average(j)
        # return reward

        min_time = float('inf')
        chosen_action = None
        for i, processing_time in enumerate(state[self.processing_time_todo]):
            if state[self.is_task_ready][i] == 1 and state[self.done_flag][i] == 0 and processing_time < min_time:
                min_time = processing_time
                chosen_action = i

        if chosen_action != action:
            return -1
        else:
            return 1

    def get_next_state(self, state, action):
        # Function to determine the next state based on the current state and action
        next_state = list()
        for s in state:
            next_state.append(list(s))

        if next_state[self.nonpreemtive_flag][action] == 0:
            next_state[self.processing_time_todo][action] = max(0, next_state[self.processing_time_todo][action] - 1)
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

        k = 0
        while True:
            for i in range(0, self.numb_of_tasks):
                if next_state[self.processing_time_todo][i] == 0 and next_state[self.lead_time_todo][i] == 0 \
                        and next_state[self.done_flag][i] == 0:
                    next_state[self.done_flag][i] = 1

            next_state[self.is_task_ready] = np.ones(self.numb_of_tasks, dtype=int)
            j = 0
            for i in next_state[self.child_foreign_keys]:
                if i != -1 and next_state[self.done_flag][j] == 0:
                    next_state[self.is_task_ready][i] = 0
                j += 1

            if len(self.get_possible_actions(next_state)[0]) != 0 \
                    or sum(next_state[self.done_flag]) == self.numb_of_tasks:
                break

            if k == 50:
                print("OH We're in a permanent loop woops! -> please report this")
                return None

            for i in range(0, self.numb_of_tasks):
                if next_state[self.done_flag][i] == 0 and next_state[self.processing_time_todo][i] == 0:
                    next_state[self.lead_time_todo][i] = max(0, next_state[self.lead_time_todo][i] - 1)
                if next_state[self.done_flag][i] == 0:
                    next_state[self.deadline][i] -= 1
            k += 1

        self.history.append((state, action))
        return next_state

    def get_possible_actions(self, state):
        # Function to determine possible actions in the current state
        possible_actions = []
        impossible_actions = []

        for action in self.actions:
            if (state[self.is_task_ready][action] == 0) or (state[self.done_flag][action] == 1) or \
                    (state[self.lead_time_todo][action] != 0 and state[self.processing_time_todo][action] == 0):
                impossible_actions.append(action)
            else:
                possible_actions.append(action)

        return possible_actions, impossible_actions

    def to_tensor_state(self, state):
        # Function to prepare the state for Deep Q-Network (DQN) processing
        state_padded = list()
        for s in state:
            s_padded = np.pad(s, (0, self.max_numb_of_tasks - len(s)), constant_values=-1)
            state_padded.append(s_padded)
        return tf.convert_to_tensor(state_padded)

    def get_result(self):
        # Function to retrieve the history
        return self.history
