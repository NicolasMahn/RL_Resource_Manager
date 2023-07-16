import numpy as np
from .generic_environment import GenericEnvironment
import util

random = np.random.random
randint = np.random.randint

class ResourceManagement(GenericEnvironment):

    def __init__(self, numb_of_machines, tasks):
        self.steps = 0
        self.tasks = tasks
        self.numb_of_tasks = len(tasks)
        self.numb_of_machines = numb_of_machines
        self.max_time = sum([task for task in tasks])
        self.max_task = max(tasks)

        start_state = np.zeros((numb_of_machines + self.numb_of_tasks,), dtype=int)
        for i in range(numb_of_machines, numb_of_machines + self.numb_of_tasks):
            start_state[i] = tasks[i-numb_of_machines]

        dimensions = np.ones((numb_of_machines + self.numb_of_tasks,), dtype=int)
        for i in range(numb_of_machines):
            dimensions[i] = self.max_task
        for i in range(numb_of_machines, numb_of_machines + self.numb_of_tasks):
            dimensions[i] = tasks[i-numb_of_machines]

        # actions[task][machine][timepoint (as int)]
        actions = []
        for task in range(0, self.numb_of_tasks):
            for machine in range(numb_of_machines):
                actions.append([task, machine])
        actions.append([-1, -1])

        super().__init__(dimensions=dimensions,
                         actions=actions,
                         start_state=start_state,
                         number_of_possible_states=util.binary_to_decimal(dimensions))
        pass
    
    def int_state_to_tuple(self, int_state):
        return util.decimal_to_binary(int_state, self.max_time)

    def done(self, state):
        if sum(state) == 0:
            return True
        else:
            False

    def extract_info_from_state(self, state):
        machines = state[:self.numb_of_machines]
        tasks = state[-self.numb_of_tasks:]

        return machines, tasks

    # since rewards are actually given for state action pairs
    def get_reward(self, state, action, next_state):
        self.steps += 1
        machines, tasks = self.extract_info_from_state(next_state)
        reward = 0

        #if action[0] == -1:
        #    reward = -5

        if any(machines) == 0 and action[0] == -1:
            # change reward depending on machine
            reward -= 5

        if self.done(next_state):
            reward = (self.max_time * 10)/self.steps
            self.steps = 0
            return reward

        #max_machine = max(machines)
        #if max_machine == 0:
        #    return -10
        # print(f"reward: {reward}")
        return reward

    def get_next_state(self, state, action):
        machines, tasks = self.extract_info_from_state(state)
        if action[0] == -1:
            for i in range(len(machines)):
                if machines[i] > 0:
                    machines[i] -= 1
        else:
            tasks[action[0]] = 0
            machines[action[1]] += self.tasks[action[0]]

        next_state = self.get_state_from_machines_and_tasks(machines, tasks)
        return next_state

    def get_possible_actions(self, state):
        possible_actions = []
        impossible_actions = []
        machines, tasks = self.extract_info_from_state(state)

        for task in range(self.numb_of_tasks):
            if tasks[task] != 0:
                possible = True
            else:
                possible = False
            j = 0
            for machine in machines:
                if machine <= 0 and possible:
                    possible_actions.append([task, j])
                else:
                    impossible_actions.append([task, j])
                j += 1

        if sum(machines) > 0:
            possible_actions.append([-1, -1])
        else:
            impossible_actions.append([-1, -1])

        return possible_actions, impossible_actions

    def get_state_from_machines_and_tasks(self, machines, tasks):
        state = np.zeros((self.numb_of_machines + self.numb_of_tasks,), dtype=int)
        for i in range(len(machines)):
            state[i] = machines[i]
        for i in range(len(machines), len(state)):
            state[i] = tasks[i- len(machines)]
        return state

    def state_for_dqn(self, state):
        return state
        #tasks = self.extract_tasks(state)
        #machines = self.extract_machines(state)
        #state = [util.binary_to_decimal(machine) for machine in machines]
        #state.append(util.binary_to_decimal(tasks))
        #state.append(self.max_time)
        #return state
        # return util.binary_to_decimal(tasks), self.max_time, util.binary_to_decimal(util.flatmap_list(machines)),
