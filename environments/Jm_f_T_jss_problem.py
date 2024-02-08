import numpy as np
import tensorflow as tf

from .generic_environment import GenericEnvironment
from resources import util, data_generation

# Random number generators
random = np.random.random
randint = np.random.randint
weighted_randint = util.weighted_randint


class Jm_f_T_JSSProblem(GenericEnvironment):

    def __init__(self, max_numb_of_tasks, max_task_depth, fixed_max_numbers,
                 high_numb_of_tasks_preference, dir_name):
        self.env_name = "[J,m=1|nowait,f,gj=1|T]"
        self.dir_name = dir_name

        # Initialize TimeManagement environment with specific parameters
        self.possible_actions = None  # Actions that can be taken in the current state
        self.impossible_actions = None  # Actions that cannot be taken
        self.fixed_max_numbers = fixed_max_numbers  # Flag to keep number of tasks constant
        self.high_numb_of_tasks_preference = high_numb_of_tasks_preference  # Preference for high number of tasks
        self.max_numb_of_tasks = max_numb_of_tasks  # Maximum number of tasks in the environment
        self.max_task_depth = max_task_depth  # Maximum depth of tasks

        # Defining dimensions for the environment
        dimensions = [2, self.max_numb_of_tasks]

        # Initialize tasks and results
        self.numb_of_tasks = self.max_numb_of_tasks
        self.tasks = np.zeros(self.numb_of_tasks)
        self.result = np.zeros(self.numb_of_tasks, dtype=int)

        # Generate possible actions
        actions = [[task, i] for task in range(self.max_numb_of_tasks) for i in range(len(self.result))]

        # Call to superclass constructor
        super().__init__(dimensions=dimensions, actions=actions, start_state=np.zeros(max_numb_of_tasks))

    def get_specific_state(self, tasks):
        # Function to get a specific state based on provided tasks
        self.tasks = tasks
        self.numb_of_tasks = len(self.tasks)
        self.result = np.zeros(self.numb_of_tasks, dtype=int)

        return list([self.tasks, self.result])

    def get_specific_state_list(self, list_):
        # Function to get a specific state based on provided tasks
        return self.get_specific_state(list_[0])

    def get_start_state(self, num_episode: int):
        # Function to initialize the starting state of the environment
        self.tasks = data_generation.get_start_state(self.env_name, self.numb_of_tasks, num_episode, self.dir_name)[0]
        self.result = np.zeros((self.numb_of_tasks,), dtype=int)
        return list([self.tasks, self.result])

    def done(self, state):
        # Function to check if the current state is a terminal state
        if sum(state[0]) == 0:
            return True
        else:
            return False

    def check_if_step_correct(self, state, action, next_state):
        merged_list = next_state[0] + next_state[1]
        sorted_state = sorted([item for item in merged_list if item != 0])
        result = next_state[1]
        i = 0
        correct = True
        for s in sorted_state:
            correct = ((s != result[i]) ^ (result[i] != 0))
            if not correct:
                return correct
            i += 1
        return correct

    def get_correct_action(self, state):
        # Action selection and masking
        possible_actions, impossible_actions = self.get_possible_actions(state)
        if len(possible_actions) == 0:
            return False

        for possible_action in possible_actions:
            if self.check_if_step_correct(state, possible_action, self.get_next_state(state, possible_action)):
                return possible_action
        return False

    def get_reward(self, state, action, next_state):
        # Function to calculate reward based on current state, action, and next state
        if action[1] - 1 < 0:
            if state[0][action[0]] == min(self.tasks):
                reward = 1
            else:
                reward = -1
        elif state[1][action[1] - 1] <= state[0][action[0]]:
            reward = 1
        else:
            reward = -1

        if action[1] + 1 >= len(state[1]):
            if state[0][action[0]] == max(self.tasks):
                reward += 1
            else:
                reward += -1
        elif state[1][action[1] + 1] >= state[0][action[0]]:
            reward += 1
        else:
            reward += -1

        return self.check_if_step_correct(state, action, next_state) #reward / 2

    def get_next_state(self, state, action):
        # Function to determine the next state based on the current state and action
        next_state = list()
        for s in state:
            next_state.append(list(s))

        next_state[1][action[1]] = next_state[0][action[0]]
        next_state[0][action[0]] = 0
        self.result = next_state[1]
        return next_state

    def get_possible_actions(self, state):
        # Function to determine possible actions in the current state
        possible_actions = []
        impossible_actions = []

        for task in range(self.max_numb_of_tasks):
            if len(state[0]) <= task:
                for i in range(self.max_numb_of_tasks):
                    impossible_actions.append([task, i])
            else:
                if state[0][task] != 0:
                    for i in range(self.max_numb_of_tasks):
                        if len(state[1]) > i:
                            if state[1][i] == 0:
                                possible_actions.append([task, i])
                            else:
                                impossible_actions.append([task, i])
                        else:
                            impossible_actions.append([task, i])
                else:
                    for i in range(self.max_numb_of_tasks):
                        impossible_actions.append([task, i])

        self.impossible_actions = impossible_actions
        self.possible_actions = possible_actions

        return possible_actions, impossible_actions

    def pad_state(self, state):
        state_padded = list()
        for s in state:
            s_padded = np.pad(s, (0, self.max_numb_of_tasks - len(s)), constant_values=-1)
            state_padded.append(s_padded)
        return state_padded

    def get_result(self):
        # Function to retrieve the result or final state of the environment
        return [r + 1 for r in self.result]
