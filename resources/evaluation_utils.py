import time
import json
import numpy as np
import os


def convert_to_serializable(data):
    """ Convert non-serializable data (like NumPy arrays) to a serializable format. """
    if isinstance(data, np.integer):
        return int(data)  # Convert np.int64 to int
    elif isinstance(data, np.floating):
        return float(data)  # Convert np.float64 to float
    if isinstance(data, np.ndarray):
        return data.tolist()  # Convert ndarray to list
    elif isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(v) for v in data]
    return data


def create_new_execution_log_dir(start_time):
    parent_dir_path = os.path.dirname(os.getcwd())
    log_dir_path = f"{parent_dir_path}/data/evaluations/{start_time}"

    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    return log_dir_path


def log_execution_details(start_time, hyperparameters, result, model_path, monitor):
    """ Logs execution details to a file and saves it in new dir. """
    # log_file = 'execution_log.json'

    monitor.stop()
    monitor.join()
    stats = monitor.get_statistics()

    execution_time = time.time() - start_time
    new_log_entry = convert_to_serializable({
        'Execution Time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        'Duration (seconds)': execution_time,
        'System Configuration': stats,
        'Hyperparameters': hyperparameters,
        'Result': result,
        'Model Path': model_path
    })

    log_dir_path = create_new_execution_log_dir(start_time)
    log_file_path = os.path.join(log_dir_path, 'execution_log.json')

    """
    # Check if the log file already exists and read it
    if os.path.isfile(log_file):
        with open(log_file, 'r') as file:
            existing_logs = json.load(file)
    else:
        existing_logs = []

    # Append the new log entry
    existing_logs.append(new_log_entry)"""

    # Write the updated logs back to the file
    with open(log_file_path, 'w') as file:
        json.dump(new_log_entry, file, indent=4)
