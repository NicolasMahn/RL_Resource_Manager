import math

import algorithms.dqn as alg
import resources.util as util

def time_dqn(test_set, env, dqn_model, less_comments):
    # Evaluates the performance of a DQN model in a time management scenario
    sortedness_list = []
    for item in test_set:
        # Calculate the optimal policy for each test set item
        _ = alg.get_pi_from_q(env, dqn_model, env.get_specific_state(item["tasks"]),
                              less_comments)
        result = env.get_result()
        # Append the sortedness percentage of the result
        sortedness_list.append(sortedness_percentage(result[0]))
    # Return the average sortedness percentage over the test set
    return sum(sortedness_list) / len(sortedness_list)

def resource_dqn(test_set, env, dqn_model, less_comments):
    # Evaluates the performance of a DQN model in a resource management scenario
    time_list = []
    optimal_time_list = []
    for item in test_set:
        # Obtain the optimal policy for each test set item
        policy = alg.get_pi_from_q(env, dqn_model, env.get_specific_state(item["tasks"], item["numb_of_machines"]),
                                   less_comments)
        # Append the time taken by the policy and the optimal time
        time_list.append(util.time_from_policy(env, policy))
        optimal_time_list.append(item["time"])
    # Calculate and return the RMSE between actual and predicted times
    return calculate_rmse(optimal_time_list, time_list)

def calculate_mse(actual, predicted):
    # Calculate the Mean Squared Error (MSE) between actual and predicted values
    if len(actual) != len(predicted):
        raise ValueError("The lengths of actual and predicted arrays should be equal.")
    squared_errors = [(actual[i] - predicted[i]) ** 2 for i in range(len(actual))]
    mse = sum(squared_errors) / len(actual)
    return mse

def calculate_rmse(actual, predicted):
    # Calculate the Root Mean Squared Error (RMSE) between actual and predicted values
    if len(actual) != len(predicted):
        raise ValueError("The lengths of actual and predicted arrays should be equal.")
    squared_errors = [(actual[i] - predicted[i]) ** 2 for i in range(len(actual))]
    mse = sum(squared_errors) / len(actual)
    rmse = math.sqrt(mse)
    return rmse

def sortedness_percentage(a):
    # Calculate the sortedness percentage of a list 'a'
    num_pairs = len(a) - 1  # total number of pairs
    if num_pairs == 0:
        return 100.0  # consider a list with one element as fully sorted
    sorted_pairs = sum(a[i] <= a[i + 1] for i in range(num_pairs))  # count of sorted pairs
    return (sorted_pairs / num_pairs) * 100  # percentage of sortedness
