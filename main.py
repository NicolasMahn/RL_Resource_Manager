import tqdm
import tensorflow as tf
import os

from resources import util, visualise_results as vis

from evaluation_monitoring import validation
from evaluation_monitoring.optimal_algorithm import generate_test_data

from environments.time_management import TimeManagement
from environments.resource_management import ResourceManagement

import algorithms.dqn as alg

# ----------------------------------------------------------------------------------------------------------------------
# Setting up GPU usage for TensorFlow:
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify GPU index for use
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow dynamic GPU memory allocation
# ----------------------------------------------------------------------------------------------------------------------

def main():
    # Check and print GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print("Using a GPU:", gpu.name)
    else:
        print("No GPUs found")
        print("GPUs found:", gpus)

    # ------------------------------------------------------------------------------------------------------------------
    # Change Hyperparameters here:

    # |Environment parameters|
    environment = "Resource"  # Choose between the "Time" or "Resource" environment
    max_numb_of_machines = 2  # Maximum number of machines. Has to be 1 if not "Resource" environment
    max_numb_of_tasks = 10  # Maximum number of tasks
    max_task_depth = 10  # duration of a task ~= random(1,max_task_depth)
    fixed_max_numbers = False
    # Toggle whether the number of tasks or machines should stay the same for each training scenario

    # |Weights for Environment Probabilities|
    # If the number of tasks or machines changes (fixed_max_numbers = False) a preference for higher tasks can be set.
    # If set to 0.25  the number of tasks will be a true random
    high_numb_of_tasks_preference = 0.35
    high_numb_of_machines_preference = 0.8  # Specific to "Resource" environment

    # |DQN algorithm parameters|
    episodes = 100  # Total number of episodes for training the DQN agent
    gamma = 0.85  # Discount factor for future rewards in the Q-learning algorithm
    epsilon = 0.4  # Initial exploration rate in the epsilon-greedy strategy
    alpha = 0.1  # Learning rate, determining how much new information overrides old information
    epsilon_decay = 0.9  # Decay rate for epsilon, reducing the exploration rate over time
    min_epsilon = 0  # Minimum value to which epsilon can decay, ensuring some level of exploration
    batch_size = 5  # Size of the batch used for training the neural network in each iteration
    update_target_network = 5  # Number of episodes after which the target network is updated

    # |Miscellaneous settings|
    numb_of_executions = 1  # The number of DQNs trained. If number is > 1 an average fitness curve will be displayed
    save_final_dqn_model = False  # Toggle to save DQN model. Only works if numb_of_executions is 1
    model_name = "TEST"  # The name the model is saved under
    test_set_abs_size = 20  # Only works if numb_of_executions is 1
    less_comments = True  # Reduce the print statements produced by the algorithm
    print_hyperparameters = True  # Toggle for printing hyperparameters

    # |Example configuration (possible only if numb_of_executions == 1)|
    # This is the example that will be displayed as an example of what the system can do
    tasks = [4, 1, 2, 3]
    numb_of_machines = 2  # Specific to "Resource" environment
    # ------------------------------------------------------------------------------------------------------------------

    # Execution logic based on the number of runs specified
    if numb_of_executions == 1:
        print("\nTraining the DQN model...")
        progress_bar = None
        # A test set is created to show the effectiveness of the algorithm
        _, test_set = generate_test_data(max_numb_of_machines, max_numb_of_tasks, max_task_depth,
                                         high_numb_of_tasks_preference, high_numb_of_machines_preference,
                                         test_set_abs_size, fixed_max_numbers)
    else:
        print("\nTraining the DQN models...")
        progress_bar = tqdm(total=numb_of_executions, unit='iterations')
        test_set = None

    fitness_curve_list = []
    results = []

    # Multiple execution loop
    if numb_of_executions > 1:
        for _ in range(numb_of_executions):

            # Environment setup based on selected type
            if environment == "Time":
                env = TimeManagement(max_numb_of_tasks, max_task_depth, test_set, fixed_max_numbers,
                                     high_numb_of_tasks_preference)
            else:
                env = ResourceManagement(max_numb_of_machines, max_numb_of_tasks, max_task_depth, fixed_max_numbers,
                                         high_numb_of_tasks_preference,
                                         high_numb_of_machines_preference, test_set)

            # Running the Q-learning algorithm
            _, result, _ = alg.q_learning(env, episodes, gamma, epsilon, alpha, epsilon_decay, min_epsilon,
                                          batch_size, update_target_network, get_pretrained_dqn=True,
                                          progress_bar=False)
            results.append(result)

            if progress_bar:
                # Update the progress bar
                progress_bar.update(1)

        # Fitness curve calculation and visualization
        for item in results:
            fitness_curve_list.append(item)

        vis.show_fitness_curve(util.calculate_average_sublist(fitness_curve_list), title="Average Fitness Curve",
                               subtitle=f"DQN average performance of {len(fitness_curve_list)} executions")

    else:  # Single execution logic

        # Environment setup based on selected type
        if environment == "Time":
            env = TimeManagement(max_numb_of_tasks, max_task_depth, test_set, fixed_max_numbers,
                                 high_numb_of_tasks_preference)
        else:
            env = ResourceManagement(max_numb_of_machines, max_numb_of_tasks, max_task_depth, fixed_max_numbers,
                                     high_numb_of_tasks_preference,
                                     high_numb_of_machines_preference, test_set)

        #  Running the Q-learning algorithm
        dqn_model, fitness_curve, pretraining_dqn_model = alg.q_learning(env, episodes, gamma, epsilon, alpha,
                                                                         epsilon_decay, min_epsilon,
                                                                         batch_size, update_target_network,
                                                                         get_pretrained_dqn=True, progress_bar=True)

        # Fitness curve calculation and visualization
        vis.show_fitness_curve(fitness_curve, title="Fitness Curve", subtitle=f"DQN")
        print("\n")

        # Environment-specific accuracy computation and visualization
        if environment == "Time":
            print(f"The accuracy of the algorithm is: {validation.time_dqn(test_set, env, dqn_model)}%")
            print("This shows the average correctly assorted tasks")
        else:
            print(f"The accuracy of the algorithm is: "
                  f"{validation.resource_dqn(test_set, env, dqn_model, less_comments)}/"
                  f"{validation.resource_dqn(test_set, env, pretraining_dqn_model, less_comments)}")
            print("The accuracy is calculated using the Root Mean Squared Error (RMSE). Lower is better.")
            print("The second number is the RMSE of the untrained NN")
        print("")

        # Example execution and visualization
        print("THE EXAMPLE ")
        if environment != "Time":
            print("The Tasks:")
            vis.visualise_tasks(tasks)
        optimal_policy = alg.get_pi_from_q(env, dqn_model, tasks, numb_of_machines, less_comments)
        print("\nThe DQN recommended policy:")
        vis.visualise_results(optimal_policy, env)
        print("\n")

        # Model saving logic
        if save_final_dqn_model:
            dqn_model.save(f'models/{model_name}')

    # Print hyperparameters if enabled
    if print_hyperparameters:
        print("environment parameters")
        print("environment:", environment)
        print("max_numb_of_machines:", max_numb_of_machines)
        print("max_numb_of_tasks:", max_numb_of_tasks)
        print("max_task_depth:", max_task_depth)
        print("fixed_max_numbers:", fixed_max_numbers)
        print("high_numb_of_tasks_preference:", high_numb_of_tasks_preference)
        print("high_numb_of_machines_preference:", high_numb_of_machines_preference)
        print("")

        print("algorithm parameters")
        print("episodes:", episodes)
        print("gamma:", gamma)
        print("epsilon:", epsilon)
        print("alpha:", alpha)
        print("epsilon_decay:", epsilon_decay)
        print("min_epsilon:", min_epsilon)
        print("batch_size:", batch_size)
        print("update_target_network:", update_target_network)
        print("")

        print("miscellaneous")
        print("numb_of_executions:", numb_of_executions)
        print("save_final_dqn_model:", save_final_dqn_model)
        print("test_set_abs_size:", test_set_abs_size)


if __name__ == '__main__':
    main()
