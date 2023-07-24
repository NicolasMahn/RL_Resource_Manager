import numpy as np

from .generic_environment import GenericEnvironment
import util
import tensorflow as tf

random = np.random.random
randint = np.random.randint
weighted_randint = util.weighted_randint


class TimeManagement(GenericEnvironment):

    def __init__(self, max_numb_of_tasks, max_task_depth, test_set, fixed_max_numbers,
                 high_numb_of_tasks_preference):
        self.possible_actions = None
        self.impossible_actions = None
        self.fixed_max_numbers = fixed_max_numbers
        self.high_numb_of_tasks_preference = high_numb_of_tasks_preference
        self.max_numb_of_tasks = max_numb_of_tasks

        self.max_task_depth = max_task_depth
        if test_set is not None:
            self.test_set = test_set
            self.test_set_tasks = [item["tasks"] for item in test_set]
        else:
            self.test_set = None
            self.test_set_tasks = None

        dimensions = [2, self.max_numb_of_tasks]

        self.numb_of_tasks = self.max_numb_of_tasks
        self.tasks = np.zeros(self.numb_of_tasks)
        self.result = np.zeros(self.numb_of_tasks, dtype=int)

        actions = []
        for task in range(self.max_numb_of_tasks):
            for i in range(len(self.result)):
                actions.append([task, i])

        super().__init__(dimensions=dimensions,
                         actions=actions,
                         start_state=np.zeros(max_numb_of_tasks))

    def get_specific_state(self, tasks, _):
        self.tasks = tasks
        self.numb_of_tasks = len(self.tasks)
        self.result = np.zeros(self.numb_of_tasks, dtype=int)

        return list([self.tasks, self.result])

    def get_start_state(self):
        self.numb_of_tasks, self.tasks = \
            util.generate_specific_time_job_shop(
                self.max_numb_of_tasks, self.max_task_depth,
                self.high_numb_of_tasks_preference, self.fixed_max_numbers,
                self.test_set_tasks)
        self.result = np.zeros((self.numb_of_tasks,), dtype=int)

        return list([self.tasks, self.result])

    def done(self, state):
        if sum(state[0]) == 0:
            return True
        else:
            return False

    # since rewards are actually given for state action pairs
    def get_reward(self, state, action, next_state):
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

        return reward/2

    def get_next_state(self, state, action):
        next_state = list()
        for s in state:
            next_state.append(list(s))

        next_state[1][action[1]] = next_state[0][action[0]]
        next_state[0][action[0]] = 0
        self.result = next_state[1]
        return next_state

    def get_possible_actions(self, state):
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

    def state_for_dqn(self, state):
        state_padded = list()
        for s in state:
            s_padded = np.pad(s, (0, self.max_numb_of_tasks - len(s)), constant_values=-1)
            state_padded.append(s_padded)
        return tf.convert_to_tensor(state_padded)

    def get_result(self):
        list_ = []
        list_.append([r + 1 for r in self.result])
        return list_
