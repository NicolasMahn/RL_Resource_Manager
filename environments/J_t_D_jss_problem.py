import numpy as np
import tensorflow as tf

from .generic_environment import GenericEnvironment
from resources import util, data_generation
from resources.reward_tracker import RewardTracker

# Random number generators
random = np.random.random
randint = np.random.randint
reward_tracker = RewardTracker()


class J_t_D_JSSProblem(GenericEnvironment):

    def __init__(self, max_numb_of_machines, max_numb_of_tasks, max_task_depth, fixed_max_numbers,
                 high_numb_of_tasks_preference, high_numb_of_machines_preference, dir_name):
        self.env_name = "[J|nowait,t,gj=1|D]"
        self.dir_name = dir_name

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

        self.done_flag = False

        # Defining dimensions and actions for the environment
        dimensions = np.array([self.max_numb_of_machines + 2, self.max_time])
        actions = [[task, machine] for task in range(1, self.max_numb_of_tasks + 1)
                   for machine in range(self.max_numb_of_machines)]
        actions.append([-1, -1])  # Adding the special action to advance time
        actions = np.array(actions)

        self.numb_of_tasks = self.max_numb_of_tasks
        self.numb_of_machines = self.max_numb_of_machines
        self.tasks = np.zeros(self.numb_of_tasks)

        # Initializing the superclass (GenericEnvironment) with defined dimensions and actions
        super().__init__(dimensions=dimensions, actions=actions,
                         start_state=np.zeros(max_numb_of_machines + max_numb_of_tasks))

    def get_specific_state(self, tasks, numb_of_machines):
        # Function to get a specific state based on tasks and number of machines
        self.numb_of_tasks = len(tasks)
        self.numb_of_machines = numb_of_machines
        self.tasks = tasks
        self.current_cumulative_machines = np.zeros(self.numb_of_machines)

        start_state = [np.pad(self.tasks, (0, self.max_time - len(self.tasks)), constant_values=0)]
        start_state.extend([[0] * self.max_time for _ in range(self.numb_of_machines)])
        start_state.append(np.zeros(self.max_time, dtype=int))
        return np.array(start_state)

    def get_specific_state_list(self, list_):
        # Function to get a specific state based on tasks and number of machines
        return self.get_specific_state(tasks=list_[0], numb_of_machines=list_[1])

    def get_start_state(self, num_episode: int):
        # Function to get the starting state of the environment

        self.tasks = data_generation.get_start_state(self.env_name, self.numb_of_tasks, num_episode, self.dir_name)[0]

        self.current_cumulative_machines = np.zeros(self.numb_of_machines)

        start_state = [np.pad(self.tasks, (0, self.max_time - len(self.tasks)), constant_values=0)]
        start_state.extend([[0] * self.max_time
                            for _ in range(self.numb_of_machines)])
        start_state.append(np.zeros(self.max_time, dtype=int))
        self.done_flag = False

        return np.array(start_state)

    def done(self, state):
        # Function to check if the current state is a terminal state
        if sum(state[0]) == 0 or self.done_flag:
            return True
        else:
            return False

    @staticmethod
    def extract_info_from_state(state):
        # Function to extract machine and task information from the current state
        tasks = state[0].copy()
        machines = state[1:-1].copy()
        step = state[-1][0]

        return machines, tasks, step

    def get_reward(self, state, action, next_state):
        # Function to calculate the reward based on the state, action, and next state
        #action_index = self.action_to_int(action)
        #possible_actions, impossible_actions = self.get_possible_actions(state)
        #pa_index = [self.action_to_int(pa) for pa in possible_actions]
        #ia_index = [self.action_to_int(ia) for ia in impossible_actions]

        machines, tasks, step = self.extract_info_from_state(state)
        next_machines, next_tasks, next_step = self.extract_info_from_state(next_state)

        legal = True

        if sum(tasks) == sum(next_tasks) and step == next_step:
            legal = False

        # sm_sum = sum([m[step] for m in machines])
        # nsm_sum = sum([nm[step] for nm in next_machines])

        s_zeros = sum([np.count_nonzero(m == 0) for m in machines])
        ns_zeros = sum([np.count_nonzero(m == 0) for m in next_machines])

        if tasks[action[0]-1] == 0 and action[0] != -1:
            legal = False

        if s_zeros != ns_zeros+tasks[action[0]-1] and action[0] != -1:
            legal = False

        if sum([m[step] for m in machines]) == 0 and action[0] == -1:
            legal = False

        if legal:
            #if action_index in ia_index:
            #    print("Action determined to be possible, although impossible")
            #    print(f"State: {state}")
            #    print(f"Action: {action}")
            #    print(f"Next State: {next_state}")

            reward = s_zeros

            # penalty for uneven distribution of tasks
            reward -= (util.current_worst(self.current_cumulative_machines) -
                       util.assumed_optimal(self.current_cumulative_machines))

            if sum(state[0]) == 0:
                reward += 100

            return reward
        else:
            # if action_index in pa_index:
            #    print("Action determined to be impossible, although possible")
            #    print(f"State: {state}")
            #    print(f"Action: {action}")
            #    print(f"Next State: {next_state}")

            self.done_flag = True
            return -100


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
        state = [tasks]
        state.extend(machines)
        state.append(np.full(self.max_time, step, dtype=int))
        return np.array(state)
