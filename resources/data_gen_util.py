import numpy as np
import pickle
import os
from datetime import datetime
import re
import resources.util as util


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


def save_labeled_data_as_pkl_file(dataset: list, epochs: int, num_tasks: int, env_name: str,
                                  unlabeled_data_dir_name: str):
    date = str(datetime.now().date())
    unlabeled_data_date = get_date_from_dir_name(unlabeled_data_dir_name)
    safe_env_name = util.make_env_name_filename_conform(env_name)
    correct_ending = ".pkl"
    filename = f"{date}_unlabeled-dir-date-{unlabeled_data_date}_epochs-{epochs}_tasks-{num_tasks}_env-{safe_env_name}"
    with open(f"{os.getcwd()}/data/test/{check_filename_ending(filename, correct_ending)}", "wb") as f:
        pickle.dump(dataset, f)
    f.close()


def read_labeled_dataset_from_pkl_file(filename: str):
    correct_ending = ".pkl"
    with open(f"{os.getcwd()}/data/test/{check_filename_ending(filename, correct_ending)}", "rb") as f:
        result = pickle.load(f)
    f.close()
    return result


def get_date_from_dir_name(unlabeled_data_dir_name: str):
    # Regular expression pattern for the date format YYYY-MM-DD
    date_pattern = r'\d{4}-\d{2}-\d{2}'

    # Search for the pattern in the directory name
    match = re.search(date_pattern, unlabeled_data_dir_name)

    # Return the matched date if found, otherwise return None (as String)
    return match.group(0) if match else "None"


def save_training_data_as_pkl_file(filename: str, result_list: list, episodes: int, num_tasks: int, repetition: bool):
    with open(get_complete_file_path(check_filename_ending(filename, ".pkl"), episodes=episodes, num_tasks=num_tasks,
                                     generating=True, repetition=repetition), "wb") as f:
        pickle.dump(result_list, f)
    f.close()


def save_training_data_as_txt_file(filename: str, result_list: list, episodes: int, num_tasks: int, repetition: bool):
    with open(get_complete_file_path(check_filename_ending(filename, ".txt"), episodes=episodes, num_tasks=num_tasks,
                                     generating=True, repetition=repetition), "w") as f:
        for item in result_list:
            f.write(str(item) + "\n")


def read_list_from_pkl_file(filename: str, dir_name: str):
    with open(get_complete_file_path(check_filename_ending(filename, ".pkl"), dir_name=dir_name), "rb") as f:
        result_list = pickle.load(f)
    f.close()

    return result_list


def check_filename_ending(filename: str, correct_ending: str):
    if filename.endswith(correct_ending):
        return filename
    else:
        return filename + correct_ending


def get_complete_file_path(filename: str = "", episodes: int = "", num_tasks: int = "", dir_name: str = "",
                           generating: bool = False, repetition: bool = False):
    if generating:
        date = datetime.now().date()
        file_path = os.getcwd() + f"/data/train/{str(date)}_episodes-{episodes}_tasks-{num_tasks}_repetition-{repetition}"
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
