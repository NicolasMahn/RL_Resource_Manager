import numpy as np
import pickle
import os
from datetime import datetime


def validate_child_elements(numb_of_tasks: int):
    sequence = np.random.choice(np.arange(-1, numb_of_tasks), size=numb_of_tasks,
                                p=[1 / 3] + [2 / (3 * numb_of_tasks)] * numb_of_tasks)

    p = 0
    for s in sequence:
        if s == -1:
            p += 1
            continue
        elif s == p:
            sequence[p] = (-1)
        elif circle_check(sequence, s, s, max_numb_recursion=numb_of_tasks * 3):
            sequence[p] = (-1)
        p += 1

    return sequence


def circle_check(sequence, position, find, max_numb_recursion=100):
    if max_numb_recursion == 0:
        return False

    if find == sequence[position]:
        return True
    elif sequence[position] == -1:
        return False
    else:
        return circle_check(sequence, sequence[position], find, (max_numb_recursion - 1))


@staticmethod
def generate_deadlines_with_target_average(numb_of_tasks, target_avg, value_range):
    # Calculate target total
    target_total = target_avg * numb_of_tasks

    # Generate initial random values
    deadlines = np.random.randint(value_range[0], value_range[1], size=numb_of_tasks)

    # Adjust the values to meet the target total
    while np.sum(deadlines) != target_total:
        # Calculate the difference between current total and target total
        diff = target_total - np.sum(deadlines)

        # Randomly select an index to adjust
        index = np.random.randint(0, numb_of_tasks)

        # Adjust the selected value, ensuring it stays within the range
        deadlines[index] = np.clip(deadlines[index] + diff, value_range[0], value_range[1])

    return deadlines.tolist()


def save_list_as_pkl_file(filename: str, result_list: list):
    with open(get_complete_file_path(check_filename_ending(filename), generating=True), "wb") as f:
        pickle.dump(result_list, f)
    f.close()


def read_list_from_pkl_file(filename: str, dir_name: str):
    with open(get_complete_file_path(check_filename_ending(filename), dir_name), "rb") as f:
        result_list = pickle.load(f)
    f.close()

    return result_list


def check_filename_ending(filename: str):
    if filename.endswith(".pkl"):
        return filename
    else:
        return filename + ".pkl"


def get_complete_file_path(filename: str, dir_name: str = "", generating: bool = False):
    if generating:
        date = datetime.now().date()
        file_path = os.getcwd() + "/data/train/" + str(date)
    else:
        file_path = os.getcwd() + "/data/train/" + dir_name
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    return file_path + "/" + filename


def remove_child_element_keys_that_are_too_high(result_list: list):
    max_key_number = result_list[0][-1]
    temp_list = result_list
    for i in range(len(temp_list[1])):
        if temp_list[1][i] > max_key_number:
            temp_list[1][i] = -1

    return temp_list