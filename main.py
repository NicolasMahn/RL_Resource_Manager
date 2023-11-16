import tqdm
from resources import util, visualise_results as vis
from environments.time_management import TimeManagement
from environments.resource_management import ResourceManagement
import algorithms.dqn as alg
import tensorflow as tf

from evaluation_monitoring.optimal_algorithm import generate_test_data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Replace "0" with the index of the GPU you want to use
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow GPU memory growth


def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print("Using a GPU:", gpu.name)
    else:
        print("No GPUs found")
        print("GPUs found:", gpus)

    ''' Change Hyperparameters here: '''

    # environment parameters
    environment = "Resource"  # Time or Resource
    max_numb_of_machines = 3  # Only for Resource environment / has to be 1 for Time environment
    max_numb_of_tasks = 10
    max_task_depth = 10
    fixed_max_numbers = True
    high_numb_of_tasks_preference = 0.35
    high_numb_of_machines_preference = 0.8  # only for Resource environment

    # algorithm parameters
    episodes = 1000
    gamma = 0.85
    epsilon = 0.4
    alpha = 0.1
    epsilon_decay = 0.9
    min_epsilon = 0
    batch_size = 32
    update_target_network = 50

    # miscellaneous
    numb_of_executions = 1  # displays average fitness if bigger than 1
    save_final_dqn_model = False  # only works if numb_of_executions is 1
    model_name = "TEST"  # the name the model is saved under
    test_set_abs_size = 50  # only works if numb_of_executions is 1
    print_hyperparameters = True

    # example (only works if numb_of_executions is 1 and has to be compatible with the environment)
    tasks = [4, 1, 2, 3]
    numb_of_machines = 2  # only for Resource environment
    '''-----------------------------'''

    if numb_of_executions == 1:
        print("\nTraining the DQN model...")
        progress_bar = None
        _, test_set = generate_test_data(max_numb_of_machines, max_numb_of_tasks, max_task_depth,
                                         high_numb_of_tasks_preference, high_numb_of_machines_preference,
                                         test_set_abs_size, fixed_max_numbers)
    else:
        print("\nTraining the DQN models...")
        progress_bar = tqdm(total=numb_of_executions, unit='iterations')
        test_set = None

    fitness_curve_list = []
    results = []
    # Run multiple threads to generate and solve job shop problems
    if numb_of_executions > 1:
        for _ in range(numb_of_executions):
            if environment == "Time":
                env = TimeManagement(max_numb_of_tasks, max_task_depth, test_set, fixed_max_numbers,
                                     high_numb_of_tasks_preference)
            else:
                env = ResourceManagement(max_numb_of_machines, max_numb_of_tasks, max_task_depth, fixed_max_numbers,
                                         high_numb_of_tasks_preference,
                                         high_numb_of_machines_preference, test_set)
            _, result, _ = alg.q_learning(env, episodes, gamma, epsilon, alpha, epsilon_decay, min_epsilon,
                                          batch_size, update_target_network, get_pretrained_dqn=True,
                                          progress_bar=False)
            results.append(result)

            if progress_bar:
                # Update the progress bar
                progress_bar.update(1)

        for item in results:
            fitness_curve_list.append(item)

        vis.show_fitness_curve(util.calculate_average_sublist(fitness_curve_list), title="Average Fitness Curve",
                               subtitle=f"DQN average performance of {len(fitness_curve_list)} executions")

    else:
        if environment == "Time":
            env = TimeManagement(max_numb_of_tasks, max_task_depth, test_set, fixed_max_numbers,
                                 high_numb_of_tasks_preference)
        else:
            env = ResourceManagement(max_numb_of_machines, max_numb_of_tasks, max_task_depth, fixed_max_numbers,
                                     high_numb_of_tasks_preference,
                                     high_numb_of_machines_preference, test_set)
        dqn_model, fitness_curve, pretrained_dqn_model = alg.q_learning(env, episodes, gamma, epsilon, alpha,
                                                                        epsilon_decay, min_epsilon,
                                                                        batch_size, update_target_network,
                                                                        get_pretrained_dqn=True, progress_bar=True)

        vis.show_fitness_curve(fitness_curve, title="Fitness Curve", subtitle=f"DQN")
        print("\n")

        if environment == "Time":
            print(f"The accuracy of the algorithm is: {util.validate_time_dqn(test_set, env, dqn_model)}%")
            # print(f"Before training it was: {validate_time_dqn(test_set, env, pretrained_dqn_model)}%")
            print("This shows the average correctly assorted tasks")
        else:
            print(f"The accuracy of the algorithm is: "
                  f"{util.validate_resource_dqn(test_set, env, dqn_model)}/"
                  f"{util.validate_resource_dqn(test_set, env, pretrained_dqn_model)}")
            print("The accuracy is calculated using the Root Mean Squared Error (RMSE). Lower is better.")
            print("The second number is the RMSE of the untrained NN")
        print("")

        print("THE EXAMPLE ")
        if environment != "Time":
            print("The Tasks:")
            vis.visualise_tasks(tasks)
        optimal_policy = alg.get_pi_from_q(env, dqn_model, tasks, numb_of_machines)
        print("\nThe DQN recommended policy:")
        vis.visualise_results(optimal_policy, env)
        print("\n")

        if save_final_dqn_model:
            dqn_model.save(f'models/{model_name}')

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
