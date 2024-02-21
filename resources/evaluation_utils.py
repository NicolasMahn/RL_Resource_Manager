import time
from datetime import datetime
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
    parent_dir_path = os.getcwd()
    log_dir_name = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d---%H:%M:%S')
    log_dir_path = f"{parent_dir_path}/data/evaluations/{log_dir_name}"

    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    return log_dir_path


def save_figures_in_log_dir(figure_list, dir_path: str):
    counter = 1
    for figure in figure_list:
        figure.savefig(f"{dir_path}/figure-{counter}")
        counter += 1


def log_execution_details(start_time, hyperparameters, result, model_path, monitor):
    """ Logs execution details to a file and saves it in new dir. """

    print(result["figures"])

    monitor.stop()
    monitor.join()
    stats = monitor.get_statistics()

    execution_time = time.time() - start_time
    new_log_entry = convert_to_serializable({
        'Execution Time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        'Duration (seconds)': execution_time,
        'System Configuration': stats,
        'Hyperparameters': hyperparameters,
        'Result': {
            'fitness_curve': result["fitness_curve"]},
        'Model Path': model_path
    })

    log_dir_path = create_new_execution_log_dir(start_time)
    log_file_path = os.path.join(log_dir_path, 'execution_log.json')

    save_figures_in_log_dir(result['figures'], log_dir_path)

    # Write the updated logs back to the file
    with open(log_file_path, 'w') as file:
        json.dump(new_log_entry, file, indent=4)
