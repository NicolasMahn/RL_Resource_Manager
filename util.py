import numpy as np
import math

random = np.random.random
randint = np.random.randint


# numpys argmax just takes the first result if there are several
# that is problematic and the reason for a new argmax func
def argmax(elements):
    # print(f"elements: {elements}")
    best_elements = argmax_multi(elements)
    return best_elements[randint(0, len(best_elements))]


def argmax_multi(elements):
    max_element = np.max(elements)
    # print(f"max_element: {max_element}")
    if math.isnan(max_element):
        raise ValueError("NaN in argmax_multi")
    best_elements = [i for i in range(len(elements)) if elements[i] == max_element]
    return best_elements


def flatmap_list(lst):
    result = []
    for item in lst:
        if isinstance(item, list) or isinstance(item, np.ndarray):
            result.extend(flatmap_list(item))
        else:
            result.append(item)
    return result


def binary_to_decimal(binary_list):
    decimal = 0
    for bit in binary_list:
        decimal = decimal * 2 + bit
    return decimal


def decimal_to_binary(decimal_number, length=None):
    binary_list = []
    while decimal_number > 0:
        bit = decimal_number % 2
        binary_list.insert(0, bit)
        decimal_number //= 2

    # Pad the binary list with leading zeros if a length is specified
    if length is not None:
        while len(binary_list) < length:
            binary_list.insert(0, 0)

    return binary_list


def weighted_randint(max_numb, high_value_preference):
    # Create an array from 0 to max_numb
    i = np.arange(max_numb + 1)

    # Adjust the preference value
    adjusted_preference = 4 * high_value_preference

    # Create weights proportional to the task number raised to adjusted high_value_preference power
    weights = np.power(i, adjusted_preference)

    # Normalize the weights so they sum to 1
    weights /= weights.sum()

    # Draw a number from the task array according to the weights
    numb = np.random.choice(i, p=weights)

    return numb


def average_difference(int_array):
    if len(int_array) < 2:   # If less than 2 elements, return None (can't compute difference)
        return 0
    diffs = [int_array[i+1]-int_array[i] for i in range(len(int_array)-1)]
    return sum(diffs) / len(diffs) # Return average difference


def calculate_average_sublist(list_of_lists):
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
    return sum(int_array) / len(int_array) # Return average difference


def current_worst(int_array):
    return max(int_array)


def current_best(int_array):
    return min(int_array)


def generate_specific_time_job_shop(max_numb_of_tasks, max_task_depth, high_numb_of_tasks_preference, fixed_max_numbers,
                                    test_set_tasks=None):
    if fixed_max_numbers:
        numb_of_tasks = max_numb_of_tasks
    else:
        numb_of_tasks = weighted_randint(max_numb_of_tasks,
                                         high_numb_of_tasks_preference)
    tasks = generate_tasks(max_task_depth, numb_of_tasks, test_set_tasks)

    return numb_of_tasks, tasks


def generate_specific_job_shop(max_numb_of_machines, max_numb_of_tasks, max_task_depth, high_numb_of_tasks_preference,
                               high_numb_of_machines_preference, fixed_max_numbers, test_set_tasks=None):
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

    return task_array.tolist()  # convert numpy array to list


def time_from_policy(env, policy):
    return max(env.current_cumulative_machines)


def find_last_nonzero_index(arr):
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] != 0:
            return i
    return 0  # Return 0 if no non-zero element is found


def calculate_mse(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("The lengths of actual and predicted arrays should be equal.")

    squared_errors = [(actual[i] - predicted[i]) ** 2 for i in range(len(actual))]
    mse = sum(squared_errors) / len(actual)
    return mse


def calculate_rmse(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("The lengths of actual and predicted arrays should be equal.")

    squared_errors = [(actual[i] - predicted[i]) ** 2 for i in range(len(actual))]
    mse = sum(squared_errors) / len(actual)
    rmse = math.sqrt(mse)
    return rmse


def is_sorted(a):
    return all(a[i] <= a[i + 1] for i in range(len(a) - 1))


def sortedness_percentage(a):
    num_pairs = len(a) - 1  # total number of pairs
    if num_pairs == 0:
        return 100.0  # consider an empty list or a list of one element as fully sorted

    sorted_pairs = sum(a[i] <= a[i + 1] for i in range(num_pairs))  # count of sorted pairs

    return (sorted_pairs / num_pairs) * 100  # percentage of sortedness
