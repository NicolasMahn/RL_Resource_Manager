import numpy as np

from .generic_environment import GenericEnvironment
from resources import util

random = np.random.random
randint = np.random.randint
weighted_randint = util.weighted_randint

class ResourceManagement(GenericEnvironment):

    def __init__(self, max_numb_of_machines, max_numb_of_tasks, max_task_depth, test_set,
                 high_numb_of_tasks_preference=0.35, high_numb_of_machines_preference=0.8):
        self.steps = 0
        self.high_numb_of_tasks_preference = high_numb_of_tasks_preference
        self.high_numb_of_machines_preference = high_numb_of_machines_preference

        self.max_numb_of_tasks = max_numb_of_tasks
        self.max_numb_of_machines = max_numb_of_machines
        self.max_time = max_numb_of_tasks*max_task_depth
        self.max_task_depth = max_task_depth
        if test_set is not None:
            self.test_set = set(test_set)
            self.test_set_tasks = set(task for task, solution in test_set)
        else:
            self.test_set = None
            self.test_set_tasks = None

        dimensions = np.concatenate([np.full(self.max_numb_of_machines, self.max_task_depth),
                               np.full(self.max_numb_of_tasks, self.max_task_depth)])

        # actions[task][machine][timepoint (as int)]
        actions = []
        for task in range(0, self.max_numb_of_tasks):
            for machine in range(self.max_numb_of_machines):
                actions.append([task, machine])
        actions.append([-1, -1])

        self.numb_of_tasks = self.max_numb_of_tasks
        self.numb_of_machines = self.max_numb_of_tasks
        self.tasks = np.zeros(self.numb_of_tasks)
        self.current_max_time = self.max_time

        super().__init__(dimensions=dimensions,
                         actions=actions,
                         start_state=np.zeros(max_numb_of_machines+max_numb_of_tasks))
        pass

    def get_specific_state(self, tasks, numb_of_machines):
        self.steps = 0
        self.numb_of_tasks = len(tasks)
        self.numb_of_machines = numb_of_machines
        self.tasks = tasks
        self.current_max_time = sum([task for task in self.tasks])

        self.start_state = np.zeros((self.numb_of_machines + self.numb_of_tasks,), dtype=int)
        for i in range(self.numb_of_machines, self.numb_of_machines + self.numb_of_tasks):
            self.start_state[i] = self.tasks[i-self.numb_of_machines]
        return list(self.start_state)

    def get_start_state(self):
        self.steps = 0
        self.numb_of_tasks = weighted_randint(self.max_numb_of_tasks,
                                              high_value_preference=self.high_numb_of_tasks_preference)
        self.numb_of_machines = weighted_randint(self.max_numb_of_machines,
                                                 high_value_preference=self.high_numb_of_machines_preference)
        self.tasks = self.generate_tasks()
        self.current_max_time = sum([task for task in self.tasks])

        self.start_state = np.zeros((self.numb_of_machines + self.numb_of_tasks,), dtype=int)
        for i in range(self.numb_of_machines, self.numb_of_machines + self.numb_of_tasks):
            self.start_state[i] = self.tasks[i-self.numb_of_machines]
        return list(self.start_state)

    def int_state_to_tuple(self, int_state):
        return util.decimal_to_binary(int_state, self.max_time)

    def done(self, state):
        if sum(state) == 0:
            return True
        else:
            return False

    def extract_info_from_state(self, state):
        machines = state[:self.numb_of_machines]
        tasks = state[-self.numb_of_tasks:]

        return machines, tasks

    # since rewards are actually given for state action pairs
    def get_reward(self, state, action, next_state):
        machines, tasks = self.extract_info_from_state(next_state)
        reward = 0

        if any(machines) == 0 and action[0] == -1:
            # change reward depending on machine
            reward = -5
            self.steps += 1

        if self.done(next_state):
            reward = (self.current_max_time * 10)/self.steps
            self.steps = 0

        # Assuming self.current_max_time * 10 is the maximum possible reward
        # max_abs_reward = max(abs(-5), abs(self.current_max_time * 10))

        # normalize reward to be between -1 and 1
        # reward = reward / max_abs_reward

        return reward

    def get_next_state(self, state, action):
        machines, tasks = self.extract_info_from_state(state)
        if action[0] == -1:
            for i in range(len(machines)):
                if machines[i] > 0:
                    machines[i] -= 1
        else:
            machines[action[1]] += tasks[action[0]]
            tasks[action[0]] = 0

        next_state = self.get_state_from_machines_and_tasks(machines, tasks)
        return next_state

    def get_possible_actions(self, state):
        self.impossible_actions = []
        self.possible_actions = []

        possible_actions = []
        impossible_actions = []
        machines, tasks = self.extract_info_from_state(state)

        for task in range(self.max_numb_of_tasks):
            if len(tasks) <= task:
                possible = False
            else:
                if tasks[task] != 0:
                    possible = True
                else:
                    possible = False

            for m in range(self.max_numb_of_machines):
                if len(machines) <= m:
                    impossible_actions.append([task, m])
                else:
                    machine = machines[m]
                    if machine <= 0 and possible:
                        possible_actions.append([task, m])
                    else:
                        impossible_actions.append([task, m])

        if any(machines) != 0:
            possible_actions.append([-1, -1])
        else:
            impossible_actions.append([-1, -1])

        self.impossible_actions = impossible_actions
        self.possible_actions = possible_actions

        return possible_actions, impossible_actions

    def get_state_from_machines_and_tasks(self, machines, tasks):
        state = np.zeros((self.numb_of_machines + self.numb_of_tasks,), dtype=int)
        for i in range(len(machines)):
            state[i] = machines[i]
        for i in range(len(machines), len(state)):
            state[i] = tasks[i- len(machines)]
        return state

    def state_for_dqn(self, state):
        m, t = self.extract_info_from_state(state)

        # Pad m and t to the desired lengths
        m_padded = np.pad(m, (0, self.max_numb_of_machines - len(m)), constant_values=-1)
        t_padded = np.pad(t, (0, self.max_numb_of_tasks - len(t)), constant_values=0)

        return np.concatenate([m_padded, t_padded])

    def generate_tasks(self):
        # Create an array from 0 to max_numb_of_tasks excluding test_set tasks
        if self.test_set_tasks is not None:
            tasks = np.array([i for i in range(self.max_task_depth + 1) if i not in self.test_set_tasks])
        else:
            tasks = np.array([i for i in range(self.max_task_depth + 1)])

        # Draw numbers from the task array randomly
        task_array = np.random.choice(tasks, size=self.numb_of_tasks)

        return task_array.tolist()  # convert numpy array to list