import numpy as np
import math
from tensorflow import keras

# Random number generators
random = np.random.random
randint = np.random.randint
random_choice = np.random.choice


def argmax(elements):
    # Custom argmax function that handles ties randomly
    best_elements = argmax_multi(elements)
    return best_elements[randint(0, len(best_elements))]


def argmax_multi(elements):
    # Finds all indices of the maximum value in elements
    max_element = np.max(elements)
    if math.isnan(max_element):
        raise ValueError("NaN in argmax_multi")
    best_elements = [i for i in range(len(elements)) if elements[i] == max_element]
    return best_elements


def flatmap_list(lst):
    # Flattens a nested list into a single list
    result = []
    for item in lst:
        if isinstance(item, list) or isinstance(item, np.ndarray):
            result.extend(flatmap_list(item))
        else:
            result.append(item)
    return result


def binary_to_decimal(binary_list):
    # Converts a binary number (list of bits) to its decimal equivalent
    decimal = 0
    for bit in binary_list:
        decimal = decimal * 2 + bit
    return decimal


def decimal_to_binary(decimal_number, length=None):
    # Converts a decimal number to its binary representation (list of bits)
    binary_list = []
    while decimal_number > 0:
        bit = decimal_number % 2
        binary_list.insert(0, bit)
        decimal_number //= 2
    if length is not None:  # Pad with zeros to match the specified length
        while len(binary_list) < length:
            binary_list.insert(0, 0)
    return binary_list


def weighted_randint(max_numb, high_value_preference):
    # Returns a weighted random integer, with higher preference for larger numbers
    i = np.arange(max_numb + 1)
    adjusted_preference = 4 * high_value_preference
    weights = np.power(i, adjusted_preference)
    weights /= weights.sum()
    numb = np.random.choice(i, p=weights)
    return numb


def average_difference(int_array):
    # Computes the average difference between consecutive elements in an array
    if len(int_array) < 2:
        return 0
    diffs = [int_array[i + 1] - int_array[i] for i in range(len(int_array) - 1)]
    return sum(diffs) / len(diffs)


def calculate_average_sublist(list_of_lists):
    # Calculates the average of each corresponding element across sublists
    if not list_of_lists:
        return None
    num_sublists = len(list_of_lists)
    sublist_length = len(list_of_lists[0])
    averages = [0] * sublist_length
    for sublist in list_of_lists:
        for i in range(sublist_length):
            if len(sublist) > i:
                averages[i] += sublist[i]
    averages = [average / num_sublists for average in averages]
    return averages


def assumed_optimal(int_array):
    # Calculates the assumed optimal (average) value from an integer array
    return sum(int_array) / len(int_array)


def current_worst(int_array):
    # Finds the maximum value in an integer array
    return max(int_array)


def current_best(int_array):
    # Finds the minimum value in an integer array
    return min(int_array)


def find_last_nonzero_index(arr):
    # Finds the index of the last non-zero element in an array
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] != 0:
            return i
    return 0  # Return 0 if no non-zero element is found


def time_from_policy(env, policy):
    # Calculates time from a given policy in an environment
    return max(env.current_cumulative_machines)


# Functions below are specific to job shop generation and handling
def generate_specific_time_job_shop(max_numb_of_tasks, max_task_depth, high_numb_of_tasks_preference, fixed_max_numbers,
                                    test_set_tasks=None):
    # Generates a specific time job shop configuration for the TimeManagement environment
    if fixed_max_numbers:
        numb_of_tasks = max_numb_of_tasks
    else:
        numb_of_tasks = weighted_randint(max_numb_of_tasks,
                                         high_numb_of_tasks_preference)
    tasks = generate_tasks(max_task_depth, numb_of_tasks, test_set_tasks)

    return numb_of_tasks, tasks


def generate_specific_job_shop(max_numb_of_machines, max_numb_of_tasks, max_task_depth, high_numb_of_tasks_preference,
                               high_numb_of_machines_preference, fixed_max_numbers, test_set_tasks=None):
    # Generates a specific time job shop configuration for the ResourceManagement environment
    if fixed_max_numbers:
        numb_of_tasks = max_numb_of_tasks
        numb_of_machines = max_numb_of_machines
    else:
        numb_of_tasks = weighted_randint(max_numb_of_tasks, high_numb_of_tasks_preference)
        numb_of_machines = weighted_randint(max_numb_of_machines, high_numb_of_machines_preference)
    tasks = generate_tasks(max_task_depth, numb_of_tasks, test_set_tasks)

    return numb_of_machines, numb_of_tasks, tasks


def generate_tasks(max_task_depth, numb_of_tasks, test_set_tasks=None):
    # Create an array from 0 to numb_of_tasks excluding test_set tasks
    if test_set_tasks is not None:
        tasks = np.array([i for i in range(1, max_task_depth + 1) if i not in test_set_tasks])
    else:
        tasks = np.array([i for i in range(1, max_task_depth + 1)])

    # Draw numbers from the task array randomly
    task_array = np.random.choice(tasks, size=numb_of_tasks)

    return task_array.tolist()  # Convert numpy array to list


def validate_child_elements(numb_of_tasks: int):
    sequence = random_choice(np.arange(-1, numb_of_tasks), size=numb_of_tasks,
                             p=[1 / 3] + [2 / (3 * numb_of_tasks)] * numb_of_tasks)

    p = 0
    for s in sequence:
        if s == -1:
            p += 1
            continue
        elif s == p:
            sequence[p] = (-1)
        elif circle_check(sequence, s, s, max_numb_recursion=numb_of_tasks*3):
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
        return circle_check(sequence, sequence[position], find, (max_numb_recursion-1))


def preprocess_data(env, dataset):
    # for i, state in enumerate(df['state']):
    #    print(f"Index {i}, Shape: {np.array(state).shape}")
    x = np.stack([d['state'] for d in dataset])

    y = np.zeros((len(dataset), len(env.actions)))  # Initialize matrix of zeros
    for i, d in enumerate(dataset):
        # Assuming d['correct_actions'] contains indices of correct actions
        for action_index in d['correct_actions']:
            y[i, action_index] = 1  # Set corresponding positions to 1
    return x, y


def make_env_name_filename_conform(env_name: str):
    # Replace characters that might cause issues in filenames
    safe_env_name = env_name.replace('|', '-')

    return safe_env_name
