import numpy as np
from .generic_environment import GenericEnvironment
from resources import util

random = np.random.random
randint = np.random.randint

class ResourceManagement(GenericEnvironment):

    def __init__(self, numb_of_machines, tasks):
        self.tasks = tasks
        self.numb_of_tasks = len(tasks)
        self.numb_of_machines = numb_of_machines

        max_time = sum([task for task in tasks])
        self.max_time = max_time

        dimensions = np.ones((max_time * numb_of_machines + self.numb_of_tasks,), dtype=int)

        # actions[task][machine][timepoint (as int)]
        actions = []
        for task in range(self.numb_of_tasks):
            for machine in range(numb_of_machines):
                for timepoint in range(max_time):
                    if max_time - timepoint < tasks[task]:
                        break
                    actions.append([task, machine, timepoint])

        super().__init__(dimensions=dimensions,
                         actions=actions,
                         start_state=np.zeros((max_time * numb_of_machines + len(tasks),), dtype=int),
                         number_of_possible_states=util.binary_to_decimal(dimensions))
        pass
    
    def int_state_to_tuple(self, int_state):
        return util.decimal_to_binary(int_state, self.max_time)

    def done(self, state):
        for task in self.extract_tasks(state):
            if task != 1:
                return False
        return True

    def extract_machines(self, state):
        machines = list(state)
        machines = machines[:-self.numb_of_tasks]
        return [machines[i:i + self.max_time] for i in range(0, len(machines), self.max_time)]

    def extract_tasks(self, state):
        return state[-self.numb_of_tasks:]

    # since rewards are actually given for state action pairs
    def get_reward(self, next_state):
        machines = self.extract_machines(next_state)
        # tasks = self.extract_tasks(next_state)
        reward = 0
        last_timepoint = 0
        for machine in machines:
            for timepoint in range(len(machine)):
                if machine[timepoint] == 1 and timepoint > last_timepoint:
                    last_timepoint = timepoint
        for machine in machines:
            for timepoint in range(last_timepoint+1):
                if machine[timepoint] == 0:
                    reward -= 1
        reward += 2*self.max_time - (last_timepoint+1)
        # print(f"reward: {reward}")
        return reward

    def get_next_state(self, state, action):
        machines = self.extract_machines(state)
        tasks = self.extract_tasks(state)
        tasks[action[0]] = 1
        for i in range(action[2], action[2]+self.tasks[action[0]]):
            machines[action[1]][i] = 1
        next_state = self.get_state_from_machines_and_tasks(machines, tasks)
        # print(len(next_state)-self.numb_of_tasks-self.numb_of_machines*self.max_time)
        return next_state

    def get_possible_actions(self, state):
        possible_actions = []
        impossible_actions = []
        tasks = self.extract_tasks(state)
        machines = self.extract_machines(state)

        for task in range(self.numb_of_tasks):
            if tasks[task] == 0:
                possible = True
            else:
                possible = False
            j = 0
            for machine in machines:
                for timepoint in range(self.max_time):
                    if self.max_time - timepoint < self.tasks[task]:
                        break
                    elif sum(1 for i in range(timepoint, timepoint+self.tasks[task]) if machine[i] != 0) == 0 \
                            and possible:
                        possible_actions.append([task, j, timepoint])
                    else:
                        impossible_actions.append([task, j, timepoint])
                j += 1

        return possible_actions, impossible_actions

    def get_state_from_machines_and_tasks(self, machines, tasks):
        state = []
        for machine in machines:
            state += machine
        state.extend(tasks)
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

