import numpy as np
import util

task_colors = colors = ["#008855", "#880033", "#550088", "#885500", "#007788", "#003388", "#338800", "#880077", "#881100", "#778800"]
square_char = "\u25A0"
machine_grey = "#333333"


def print_hex_color(hex_color, text):
    print(f"\033[38;2;{int(hex_color[1:3], 16)};"
          f"{int(hex_color[3:5], 16)};"
          f"{int(hex_color[5:7], 16)}m{text}\033[0m", end='')


def visualise_tasks(tasks):
    i = 0
    for task in tasks:
        print(f"Task {i}: [", end='')
        for j in range(task):
            print_hex_color(task_colors[i], square_char)
        print(f"] ({task})")
        i += 1


def visualise_machines(numb_machines, max_time):
    for machine in range(numb_machines):
        print(f"Machine {machine}[", end='')
        for timepoint in range(max_time):
            print_hex_color(machine_grey, square_char)
        print("]")


def visualise_results(optimal_policy, env):
    tasks = env.tasks
    numb_machines = env.numb_of_machines
    max_time = env.max_time

    machine = np.zeros(max_time, dtype=int)

    machines = []
    for i in range(numb_machines):
        machines.append(machine.copy())


    j = 0

    for (state, action) in optimal_policy:
        if action is not None:
            if action[0] != -1:
                for i in range(j, j + tasks[action[0]]):
                    machines[action[1]][i] = action[0]+1
            else:
                j += 1


    i = 0
    for machine in machines:
        print(f"Machine {i}: [", end='')
        for timepoint in machine:
            if timepoint == -1 or timepoint == 0:
                print_hex_color(machine_grey, square_char)
            else:
                print_hex_color(task_colors[timepoint-1], square_char)
        print("]  [", end='')
        for timepoint in machine:
            if timepoint == 0:
                print("-", end='')
            else:
                print(timepoint-1, end='')
        print("]")
        i += 1


