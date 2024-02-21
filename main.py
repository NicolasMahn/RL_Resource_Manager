import time
import json
import tqdm
import tensorflow as tf
import numpy as np
import os

from resources import util, visualise_results as vis

from evaluation_monitoring import validation
from evaluation_monitoring.optimal_algorithm import generate_test_data

from environments.Jm_f_T_jss_problem import Jm_f_T_JSSProblem
from environments.J_t_D_jss_problem import J_t_D_JSSProblem
from environments.Jm_tf_T_jss_problem import Jm_tf_T_JSSProblem

from algorithms.dqn import dqn
from algorithms.double_dqn import ddqn
from algorithms.prioritized_double_dqn import prioritized_ddqn
from algorithms.dueling_double_dqn import dueling_ddqn
from algorithms.a2c import a2c
from algorithms.supervised import supervised_learning

import resources.data_generation as data_gen
from resources.performance_monitor import PerformanceMonitor
import resources.evaluation_utils as evaluation_utils

# ----------------------------------------------------------------------------------------------------------------------
# Setting up GPU usage for TensorFlow:
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify GPU index for use
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow dynamic GPU memory allocation


# ----------------------------------------------------------------------------------------------------------------------


def setup_gpu():
    """ Sets up GPU for TensorFlow and prints its availability. """
    # Check and print GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print("Using a GPU:", gpu.name)
    else:
        print("No GPUs found")


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


def log_execution_details(start_time, hyperparameters, result, model_path, monitor):
    """ Logs execution details to a file. """
    log_file = 'execution_log.json'

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

    # Check if the log file already exists and read it
    if os.path.isfile(log_file):
        with open(log_file, 'r') as file:
            existing_logs = json.load(file)
    else:
        existing_logs = []

    # Append the new log entry
    existing_logs.append(new_log_entry)

    # Write the updated logs back to the file
    with open(log_file, 'w') as file:
        json.dump(existing_logs, file, indent=4)


def get_env_from_name(env_name: str, max_numb_of_machines: int, max_numb_of_tasks: int, max_task_depth: int,
                      fixed_max_numbers: int, high_numb_of_machines_preference: float,
                      high_numb_of_tasks_preference: float, training_dir_name: str):
    if env_name == "[J,m=1|nowait,f,gj=1|T]":
        env = Jm_f_T_JSSProblem(max_numb_of_tasks, max_task_depth, fixed_max_numbers,
                                high_numb_of_tasks_preference, training_dir_name)
    elif env_name == "[J,m=1|pmtn,nowait,tree,nj,t,f,gj=1|T]":
        env = Jm_tf_T_JSSProblem(max_numb_of_tasks, fixed_max_numbers, high_numb_of_tasks_preference,
                                 training_dir_name)
    else:
        env = J_t_D_JSSProblem(max_numb_of_machines, max_numb_of_tasks, max_task_depth, fixed_max_numbers,
                               high_numb_of_tasks_preference,
                               high_numb_of_machines_preference, training_dir_name)

    return env


def execute_algorithm(algorithm: str, env, episodes: int, epochs: int, batch_size: int, numb_of_executions: int,
                      gamma: float = 0.0, epsilon: float = 0.0, alpha: float = 0.0, epsilon_decay: float = 0.0,
                      min_epsilon: float = 0.0, rb_alpha: float = 0.0, rb_beta: float = 0.0, rb_beta_end: float = 0.0,
                      update_target_network: int = 0):
    # Algorithm based on chosen algorithm
    if algorithm == 'Supervised':
        #  Running the Supervised algorithm
        model, history, pretrained_model = supervised_learning(env, episodes, epochs, alpha, batch_size,
                                                               get_pretrained_dnn=True)

        return {
            "training_loss": history.history['loss'],
            "validation_loss": history.history.get('val_loss', []),
            "training_accuracy": history.history['accuracy'],
            "validation_accuracy": history.history.get('val_accuracy', [])
        }, model, pretrained_model
    elif algorithm == "DDQN":
        #  Running the Double Q-learning algorithm
        model, fitness_curve, pretrained_model = ddqn(env, episodes, gamma, epsilon, alpha, epsilon_decay,
                                                      min_epsilon, batch_size, update_target_network,
                                                      get_pretrained_dqn=True, progress_bar=(numb_of_executions == 1))
        return {"fitness_curve": fitness_curve}, model, pretrained_model
    elif algorithm == "Prioritized DDQN":
        #  Running the Prioritized Double Q-learning algorithm
        model, fitness_curve, pretrained_model = prioritized_ddqn(env, episodes, gamma, epsilon, alpha,
                                                                  epsilon_decay, min_epsilon, rb_alpha, rb_beta,
                                                                  rb_beta_end, batch_size, update_target_network,
                                                                  get_pretrained_dqn=True,
                                                                  progress_bar=(numb_of_executions == 1))
        return {"fitness_curve": fitness_curve}, model, pretrained_model
    elif algorithm == "Dueling DDQN":
        #  Running the Prioritized Double Q-learning algorithm
        model, fitness_curve, pretrained_model = dueling_ddqn(env, episodes, gamma, epsilon, alpha,
                                                              epsilon_decay, min_epsilon, batch_size,
                                                              update_target_network, get_pretrained_dqn=True,
                                                              progress_bar=(numb_of_executions == 1))
        return {"fitness_curve": fitness_curve}, model, pretrained_model
    elif algorithm == "A2C":
        #  Running the A2C algorithm
        actor_model, critic_model, fitness_curve = a2c(env, episodes, gamma, alpha,
                                                       progress_bar=(numb_of_executions == 1))
        return {"fitness_curve": fitness_curve}, None, None
    else:
        #  Running the Q-learning algorithm
        model, fitness_curve, pretrained_model = dqn(env, episodes, gamma, epsilon, alpha, epsilon_decay,
                                                     min_epsilon, batch_size, update_target_network,
                                                     get_pretrained_dqn=True, progress_bar=(numb_of_executions == 1))

        return {"fitness_curve": fitness_curve}, model, pretrained_model


def evaluate_results(env, numb_of_executions: int, algorithm: str, environment: str, result: dict, test_dir_name: str,
                     pretrained_model, model):
    print("\nTraining is done displaying some exploratory evaluation results")
    if numb_of_executions > 1:

        # Result calculation and visualization
        if algorithm == 'Supervised':
            training_loss_list = list()
            training_accuracy_list = list()
            for item in result:
                training_accuracy_list.append(item["training_accuracy"])
                training_loss_list.append(item["training_loss"])

            vis.show_one_line_graph(util.calculate_average_sublist(training_loss_list),
                                    title="Average Training Loss",
                                    subtitle=f"of {len(training_loss_list)} executions using the Supervised Approach",
                                    x_label="epochs", y_label="loss", start_with_zero=False)
            vis.show_one_line_graph(util.calculate_average_sublist(training_accuracy_list),
                                    title="Average Training Accuracy",
                                    subtitle=f"of {len(training_accuracy_list)} executions using the Supervised "
                                             f"Approach", x_label="epochs", y_label="accuracy")
        else:
            fitness_curve_list = list()
            for item in result:
                fitness_curve_list.append(item["fitness_curve"])

            vis.show_one_line_graph(util.calculate_average_sublist(fitness_curve_list), title="Average Fitness Curve",
                                    subtitle=f"{algorithm} average performance of {len(fitness_curve_list)}"
                                             f" executions")

    else:
        if algorithm == 'Supervised':
            vis.show_one_line_graph(result["training_loss"], title="Training Loss", subtitle=f"Supervised Approach",
                                    x_label="epochs", y_label="loss")
            vis.show_one_line_graph(result["training_accuracy"], title="Training Accuracy",
                                    subtitle=f"Supervised Approach", x_label="epochs", y_label="accuracy")

        else:
            poly_fc = vis.get_polynomial_fitness_curve(result["fitness_curve"], 10)
            vis.show_line_graph([result["fitness_curve"], poly_fc], ["Fitness Curve", "Regressed Fitness Curve"],
                                title="Fitness Curve", subtitle=algorithm)
            print("\n")

        # Environment-specific accuracy computation and visualization
        if environment == "[J,m=1|nowait,f,gj=1|T]" and algorithm != "Dueling DDQN" and algorithm != "A2C":
            loss, accuracy = validation.get_test_loss_and_accuracy(test_dir_name, env, model)
            # TODO: Accuracy is somewhat false as not all but only one correct action is selected.
            # There are and can be multiple correct actions
            pretrained_loss, pretrained_accuracy = validation.get_test_loss_and_accuracy(test_dir_name, env,
                                                                                         pretrained_model)
            print(
                f"The accuracy of the model is: {100 * accuracy}% (pretrained model had {100 * pretrained_accuracy}%)")
            print("This shows the average correctly assorted tasks")
            result["test_accuracy"] = accuracy
            result["pretrained_test_accuracy"] = pretrained_accuracy

            print(f"The mean squared error of the model is: {loss} (pretrained model had {pretrained_loss})")
            print("Lower is better.")
            result["test_loss"] = loss
            result["test_mse"] = loss
            result["pretrained_test_loss"] = pretrained_loss
            result["pretrained_test_mse"] = pretrained_loss

        """
        print("Here an example is visualised ")
        if environment == "[J,m=1|nowait,f,gj=1|T]":
            print("The Tasks:")
            vis.visualise_tasks(tasks)
        optimal_policy = dqn.get_pi_from_q(env, model, env.get_specific_state_list([tasks, numb_of_machines]),
                                           less_comments)
        print("\nThe Model recommended policy:")
        vis.visualise_results(optimal_policy, env)
        print("\n")
        """
    return result


def main():
    # Starting monitor tools for the log file
    start_time = time.time()
    monitor = PerformanceMonitor(interval=1)
    monitor.start()

    setup_gpu()

    # ------------------------------------------------------------------------------------------------------------------
    # Change Hyperparameters here:

    # |Environment Name|
    # The environments are named after German job shop scheduling classification standards.
    # Standards defined here: https://de.wikipedia.org/wiki/Klassifikation_von_Maschinenbelegungsmodellen
    # Sofar these environments have been implemented:
    # [J|nowait,t,gj=1|D]
    # [J,m=1|nowait,f,gj=1|T]
    # [J,m=1|pmtn,nowait,tree,nj,t,f,gj=1|T]
    environment = "[J|nowait,t,gj=1|D]"

    # |Environment parameters|
    max_numb_of_machines = 3  # Maximum number of machines. Has to be 1 if m=1 for the environment
    max_numb_of_tasks = 9  # Maximum number of tasks -> check that dataset has enough entries, else create new one
    max_task_depth = 10  # duration of a task ~= random(1,max_task_depth)
    fixed_max_numbers = True  # CURRENTLY ONLY TRUE POSSIBLE
    # Toggle whether the number of tasks or machines should stay the same for each training scenario

    # |Weights for Environment Probabilities|
    # If the number of tasks or machines changes (fixed_max_numbers = False) a preference for higher tasks can be set.
    # If set to 0.25  the number of tasks will be a true random
    high_numb_of_tasks_preference = 0.35
    high_numb_of_machines_preference = 0.8  # Specific to environment with more than one machine

    # |Choose Algorithm|
    # Choose between 'Supervised', 'DQN', 'DDQN', 'Prioritized DDQN' and 'Dueling DDQN', 'A2C'
    algorithm = 'A2C'

    # |DQN algorithm parameters|
    episodes = 20  # Total number of episodes for training the DQN agent
    epochs = 1  # The number of times every episode should be 'retrained' | with dqn it can only be 1
    gamma = 0.85  # Discount factor for future rewards in the Q-learning algorithm
    epsilon = 1  # Initial exploration rate in the epsilon-greedy strategy
    alpha = 0.0001  # Learning rate, determining how much new information overrides old information
    epsilon_decay = 0.995  # Decay rate for epsilon, reducing the exploration rate over time
    min_epsilon = 0.01  # Minimum value to which epsilon can decay, ensuring some level of exploration
    batch_size = 128  # Size of the batch used for training the neural network in each iteration
    update_target_network = 100  # Number of episodes after which the target network is updated
    # Parameters for the Prioritizing Replay Buffer:
    rb_alpha = 0.6  # This parameter controls how much prioritization is used
    rb_beta = 0.4  # This parameter is used for adjusting the importance-sampling weights
    rb_beta_end = 1  # The final value of β at the end of training

    # |Miscellaneous settings|
    numb_of_executions = 1  # The number of DQNs trained. If number is > 1 an average fitness curve will be displayed
    save_final_dqn_model = False  # Toggle to save DQN model. Only works if numb_of_executions is 1
    model_name = "auto"  # The name the model is saved under
    print_hyperparameters = False  # Toggle for printing hyperparameters
    save_log_file = True

    # Specify which training data should be used
    training_dir_name = "2024-02-04_episodes-690000_tasks-100"

    # Specify which test data should be used
    test_dir_name = "2024-02-20_unlabeled-dir-date-2024-01-17_epochs-1000_tasks-9_env-[J,m=1-nowait,f,gj=1-T]"
    # ------------------------------------------------------------------------------------------------------------------

    env = None
    model = None
    pretrained_model = None

    result = list()
    model_path = "No Model was saved"

    # Multiple execution loop
    if numb_of_executions > 1:
        print(f"\nTraining {numb_of_executions} {algorithm} models with the {environment} environment...")
        progress_bar = tqdm(total=numb_of_executions, unit='iterations')

    else:  # Single execution logic
        print(f"\nTraining the {algorithm} model with the {environment} environment...")
        progress_bar = None

    for _ in range(numb_of_executions):

        # Environment setup based on selected type
        env = get_env_from_name(environment, max_numb_of_machines, max_numb_of_tasks, max_task_depth, fixed_max_numbers,
                                high_numb_of_machines_preference, high_numb_of_tasks_preference, training_dir_name)

        # function to generate training data----------------------------------------------------------------------------
        # data_gen.generate_new_dataset(5000, 100, 30, True)

        # function to generate test data / labeled data
        # uses current training data to generate test data
        # the same training and test data should not be used in combination
        # number_of_tasks and environment variables should not be changed/need to be changed above
        # SOFAR ONLY WORKS FOR [J,m=1|nowait,f,gj=1|T] ENVIRONMENT
        # data_gen.label_training_data(env, epochs=1000, number_of_tasks=max_numb_of_tasks, env_name=environment,
        #                              unlabeled_data_dir_name="2024-01-17_epochs-1000_tasks-9")

        # --------------------------------------------------------------------------------------------------------------

        result_item, model, pretrained_model = execute_algorithm(algorithm, env, episodes, epochs, batch_size,
                                                                 numb_of_executions, gamma, epsilon, alpha,
                                                                 epsilon_decay, min_epsilon, rb_alpha, rb_beta,
                                                                 rb_beta_end, update_target_network)

        if numb_of_executions > 1:
            result.append(result_item)

            if progress_bar:
                # Update the progress bar
                progress_bar.update(1)
        else:
            result = result_item

    result = evaluate_results(env, numb_of_executions, algorithm, environment, result, test_dir_name,
                              pretrained_model, model)

    if numb_of_executions == 1 and algorithm != "A2C":
        # Model saving logic
        if model_name == "auto":
            save_env_name = util.make_env_name_filename_conform(environment)
            model_name = f"{save_env_name}_{algorithm}_{time.strftime('%Y%m%d_%H%M%S')}"
        model_path = f'models/{model_name}'
        model.save(model_path)

    hyperparameters = {
        'environment': environment,
        'max_numb_of_machines': max_numb_of_machines,
        'max_numb_of_tasks': max_numb_of_tasks,
        'max_task_depth': max_task_depth,
        'fixed_max_numbers': fixed_max_numbers,
        'high_numb_of_tasks_preference': high_numb_of_tasks_preference,
        'high_numb_of_machines_preference': high_numb_of_machines_preference,
        'algorithm': algorithm,
        'episodes': episodes,
        'epochs': epochs,
        'gamma': gamma,
        'epsilon': epsilon,
        'alpha': alpha,
        'epsilon_decay': epsilon_decay,
        'min_epsilon': min_epsilon,
        'rb_alpha': rb_alpha,
        'rb_beta': rb_beta,
        'rb_beta_end': rb_beta_end,
        'batch_size': batch_size,
        'update_target_network': update_target_network,
        'numb_of_executions': numb_of_executions,
        'model_name': model_name,
        'save_final_dqn_model': save_final_dqn_model,
        'save_log_file': save_log_file,
        'training_dir_name': training_dir_name,
        'test_dir_name': test_dir_name
    }

    # Print hyperparameters if enabled
    if print_hyperparameters:
        print(hyperparameters)

    # Save all sorts of details about the execution in the log file
    if save_log_file:
        log_execution_details(start_time, hyperparameters, result, model_path, monitor)
        evaluation_utils.log_execution_details(start_time, hyperparameters, result, model_path, monitor)
        print("\nLog file successfully updated")


if __name__ == '__main__':
    main()
