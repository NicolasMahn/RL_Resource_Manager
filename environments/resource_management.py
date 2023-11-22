import numpy as np
import tensorflow as tf

from .generic_environment import GenericEnvironment
from resources import util
from resources.reward_tracker import RewardTracker

# Random number generators
random = np.random.random
randint = np.random.randint
reward_tracker = RewardTracker()


class ResourceManagement(GenericEnvironment):

    def __init__(self, max_numb_of_machines, max_numb_of_tasks, max_task_depth, fixed_max_numbers,
                 high_numb_of_tasks_preference, high_numb_of_machines_preference, test_set):
        # Initialization of the Resource Management Environment
        self.machines = None  # Current state of the machines
        self.possible_actions = None  # Actions that can be taken
        self.impossible_actions = None  # Actions that cannot be taken
        self.current_cumulative_machines = None  # Cumulative state of machines

        # Setting parameters for the environment
        self.fixed_max_numbers = fixed_max_numbers
        self.high_numb_of_tasks_preference = high_numb_of_tasks_preference
        self.high_numb_of_machines_preference = high_numb_of_machines_preference
        self.max_numb_of_tasks = max_numb_of_tasks
        self.max_numb_of_machines = max_numb_of_machines
        self.max_time = max_numb_of_tasks * max_task_depth
        self.max_task_depth = max_task_depth

        # Handling test sets if provided
        self.test_set = test_set
        self.test_set_tasks = [item["tasks"] for item in test_set] if test_set is not None else None

        # Defining dimensions and actions for the environment
        dimensions = [self.max_numb_of_machines + 2, self.max_time]
        actions = [[task, machine] for task in range(1, self.max_numb_of_tasks + 1)
                   for machine in range(self.max_numb_of_machines)]
        actions.append([-1, -1])  # Adding the special action to advance time

        self.numb_of_tasks = self.max_numb_of_tasks
        self.numb_of_machines = self.max_numb_of_machines
        self.tasks = np.zeros(self.numb_of_tasks)
        self.current_max_time = self.max_time

        # Initializing the superclass (GenericEnvironment) with defined dimensions and actions
        super().__init__(dimensions=dimensions, actions=actions,
                         start_state=np.zeros(max_numb_of_machines + max_numb_of_tasks))


    def get_specific_state(self, tasks, numb_of_machines):
        # Function to get a specific state based on tasks and number of machines
        self.numb_of_tasks = len(tasks)
        self.numb_of_machines = numb_of_machines
        self.tasks = tasks
        self.current_max_time = sum([task for task in self.tasks])
        self.current_cumulative_machines = np.zeros(self.numb_of_machines)

        start_state = [np.pad(self.tasks, (0, self.current_max_time - len(self.tasks)), constant_values=0)]
        start_state.extend([[0] * self.current_max_time for _ in range(self.numb_of_machines)])
        start_state.append(np.zeros(self.current_max_time, dtype=int))
        return list(start_state)

    def get_start_state(self):
        # Function to get the starting state of the environment
        self.numb_of_machines, self.numb_of_tasks, self.tasks = \
            util.generate_specific_job_shop(
                self.max_numb_of_machines, self.max_numb_of_tasks,
                self.max_task_depth,
                self.high_numb_of_tasks_preference,
                self.high_numb_of_machines_preference,
                self.fixed_max_numbers, self.test_set_tasks)

        self.current_cumulative_machines = np.zeros(self.numb_of_machines)
        self.current_max_time = sum([task for task in self.tasks])

        start_state = [np.pad(self.tasks, (0, self.current_max_time - len(self.tasks)), constant_values=0)]
        start_state.extend([[0] * self.current_max_time
                            for _ in range(self.numb_of_machines)])
        start_state.append(np.zeros(self.current_max_time, dtype=int))
        return list(start_state)

    def done(self, state):
        # Function to check if the current state is a terminal state
        if sum(state[0]) == 0:
            return True
        else:
            return False

    @staticmethod
    def extract_info_from_state(state):
        # Function to extract machine and task information from the current state
        tasks = list(state[0])
        m = list(state[1:-1])
        machines = []
        for machine in m:
            machines.append(list(machine))
        step = state[-1][0]

        return machines, tasks, step

    def get_reward(self, state, action, next_state):
        # Function to calculate the reward based on the state, action, and next state
        reward = 0.1  # small reward for each step

        # penalty for uneven distribution of tasks
        reward -= (util.current_worst(self.current_cumulative_machines) -
                   util.assumed_optimal(self.current_cumulative_machines))

        # reward += util.current_best(self.current_cumulative_machines)

        # reward = np.clip(reward/10, -1.0, 1.0)

        return reward

    def get_next_state(self, state, action):
        # Function to determine the next state based on the current state and action
        machines, tasks, step = self.extract_info_from_state(state)
        if action[0] == -1:
            step += 1
        else:
            self.current_cumulative_machines[action[1]] += \
                tasks[action[0] - 1]
            for i in range(step, tasks[action[0] - 1] + step):
                machines[action[1]][i] += action[0]
            tasks[action[0] - 1] = 0
            self.machines = machines
        next_state = self.get_state_from_machines_and_tasks(
            machines, tasks, step)
        return next_state

    def get_possible_actions(self, state):
        # Function to determine possible actions in the current state
        possible_actions = []
        impossible_actions = []
        machines, tasks, step = self.extract_info_from_state(state)

        for task in range(1, self.max_numb_of_tasks + 1):
            if len(tasks) + 1 <= task:
                possible = False
            else:
                if tasks[task - 1] != 0:
                    possible = True
                else:
                    possible = False

            for m in range(self.max_numb_of_machines):
                if len(machines) <= m:
                    impossible_actions.append([task, m])
                else:
                    machine = machines[m]
                    if machine[step] <= 0 and possible:
                        possible_actions.append([task, m])
                    else:
                        impossible_actions.append([task, m])

        if any(machine[step] != 0 for machine in machines):
            possible_actions.append([-1, -1])
        else:
            impossible_actions.append([-1, -1])

        self.impossible_actions = impossible_actions
        self.possible_actions = possible_actions

        return possible_actions, impossible_actions

    def get_state_from_machines_and_tasks(self, machines, tasks, step):
        # Function to get the current state from machine and task statuses
        start_state = [tasks]
        start_state.extend(machines)
        start_state.append(np.full(self.current_max_time, step, dtype=int))
        return list(start_state)

    def state_for_dqn(self, state):
        # Function to prepare the state for Deep Q-Network (DQN) processing
        machines, tasks, step = self.extract_info_from_state(state)

        t_padded = np.pad(tasks, (0, self.max_time - len(tasks)), constant_values=0)

        ms_padded = []
        for m in machines:
            m_padded = np.pad(m, (0, self.max_time - len(m)), constant_values=0)
            ms_padded.append(m_padded)

        while len(ms_padded) < self.max_numb_of_machines:
            ms_padded.append([-1] * self.max_time)

        full_state = [list(t_padded)]
        full_state.extend(ms_padded)
        full_state.append([int(step)] * self.max_time)
        return tf.convert_to_tensor(full_state)

    def get_result(self):
        # Function to retrieve the result or final state of the environment
        return self.machines
