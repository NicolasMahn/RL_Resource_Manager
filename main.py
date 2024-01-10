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

import algorithms.dqn as dqn
import algorithms.supervised as supervised

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

    # |Environment Name|
    # The environments are named after German job shop scheduling classification standards.
    # Standards defined here: https://de.wikipedia.org/wiki/Klassifikation_von_Maschinenbelegungsmodellen#Literatur
    # Under this standard, the job shop problem is first divided into 3 classifications:
    # α = Machine characteristics
    # β = Task characteristics
    # γ = Objective
    # These classifications are defined by several characteristics (° means "nothing"):
    # α| Machine characteristics
    # # α1: Machine type and order
    # # # °: A single available machine
    # # # IP: Identical parallel machines
    # # # UP: Uniform parallel machines (with different production speeds)
    # # # F: Flow-Shop
    # # # J: Job-Shop
    # # # O: Open-Shop
    # # α2: Number of machines
    # # # °: Any number
    # # # m: Exactly m machines
    # β| Task characteristics
    # # β1: Number of tasks
    # # # n=const: A certain number of tasks is predefined. Often n=2.
    # # # °: Any number
    # # β2: Interruptibility
    # # # pmtn: Interruption is possible (Preemption)
    # # # °: No interruption
    # # # nowait: After completing a task, the next task must start immediately.
    # # β3: Sequence relationship
    # # # prec: Predetermined sequences in the form of a graph
    # # # tree: Graph in the form of a tree
    # # # °: No sequence relationships
    # # β4: Release time and lead time
    # # # aj: Different task release times
    # # # nj: Lead times are given. After completing a task, the task must wait before it can be processed further.
    # # # °: All tasks are available from the beginning, and there are no lead times
    # # β5: Processing time
    # # # t: refers to the duration of the processing time of the entire task or the individual tasks
    # # # °: Any processing times
    # # β6: Sequence-dependent setup times
    # # # τ: Sequence-dependent setup time from task j to task k on machine i
    # # # τb: The tasks can be grouped into families
    # # # °: No sequence-dependent setup times
    # # β7: Resource constraints
    # # # res λσρ
    # # # # λ: Number of resources
    # # # # σ: Availability of resources
    # # # # ρ: Demand for resources
    # # # °: No resource constraints
    # # β8: Completion deadlines
    # # # f: Strict deadlines are given for each task
    # # # °: No deadlines given
    # # β9: Number of operations
    # # # g: Each task consists of exactly/at most n operations
    # # # °: Any number of operations
    # β10: Storage constraints
    # # # κ: Indicates the available intermediate storage for the i-th machine
    # # # °: Each machine has a storage with infinite capacity
    # γ| Objective
    # # D: Minimization of throughput time
    # # Z: Minimization of cycle time / total processing time
    # # T: Minimization of deadline deviation
    # # V: Minimization of tardiness
    # # L: Minimization of idle time
    # Sofar these environments have been implemented:
    # [J|nowait,t,gj=1|min(D)] As it is a Job Shop in which the processing time of task, of which only the processing
    #                     time is known, has to be minimized.
    # [J,m=1|nowait,f,gj=1|min(T)]
    # [J,m=1|pmtn,nowait,tree,nj,t,f,gj=1|avg(T)] # In Progress may be buggy
    environment = "[J,m=1|nowait,f,gj=1|min(T)]"  # Choose between the "[J,m=1|nowait,f|min(T)]", or
    #                                                          "[J|nowait,t|min(D)] " environment

    # |Environment parameters|
    max_numb_of_machines = 2  # Maximum number of machines. Has to be 1 if not "Resource" environment
    max_numb_of_tasks = 9  # Maximum number of tasks
    max_task_depth = 10  # duration of a task ~= random(1,max_task_depth)
    fixed_max_numbers = False
    # Toggle whether the number of tasks or machines should stay the same for each training scenario

    # |Weights for Environment Probabilities|
    # If the number of tasks or machines changes (fixed_max_numbers = False) a preference for higher tasks can be set.
    # If set to 0.25  the number of tasks will be a true random
    high_numb_of_tasks_preference = 0.35
    high_numb_of_machines_preference = 0.8  # Specific to "Resource" environment

    # |Choose Algorithm|
    # Choose between 'supervised' and 'dqn'
    algorithm = 'supervised'

    # |DQN algorithm parameters|
    episodes = 500  # Total number of episodes for training the DQN agent
    gamma = 0.85  # Discount factor for future rewards in the Q-learning algorithm
    epsilon = 0.4  # Initial exploration rate in the epsilon-greedy strategy
    alpha = 0.01  # Learning rate, determining how much new information overrides old information
    epsilon_decay = 0.9  # Decay rate for epsilon, reducing the exploration rate over time
    min_epsilon = 0  # Minimum value to which epsilon can decay, ensuring some level of exploration
    batch_size = 32  # Size of the batch used for training the neural network in each iteration
    update_target_network = 50  # Number of episodes after which the target network is updated

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
            if environment == "[J,m=1|nowait,f,gj=1|min(T)]":
                env = Jm_f_T_JSSProblem(max_numb_of_tasks, max_task_depth, test_set, fixed_max_numbers,
                                        high_numb_of_tasks_preference)
            elif environment == "[J,m=1|pmtn,nowait,tree,nj,t,f,gj=1|avg(T)]":
                env = Jm_tf_T_JSSProblem(max_numb_of_tasks, test_set, fixed_max_numbers, high_numb_of_tasks_preference)
            else:
                env = J_t_D_JSSProblem(max_numb_of_machines, max_numb_of_tasks, max_task_depth, fixed_max_numbers,
                                       high_numb_of_tasks_preference,
                                       high_numb_of_machines_preference, test_set)

            if algorithm == 'supervised':
                _, history, _ = supervised.supervised_learning(env, episodes, batch_size, get_pretrained_dnn=True)

                # Extracting metrics from the history object as arrays
                training_loss = history.history['loss']
                validation_loss = history.history.get('val_loss', [])  # Empty list if validation loss is not available
                training_accuracy = history.history['accuracy']
                validation_accuracy = history.history.get('val_accuracy',
                                                          [])  # Empty list if validation accuracy is not available
                result = training_accuracy

            else:
                # Running the Q-learning algorithm
                _, result, _ = dqn.q_learning(env, episodes, gamma, epsilon, alpha, epsilon_decay, min_epsilon,
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
        if environment == "[J,m=1|nowait,f,gj=1|min(T)]":
            env = Jm_f_T_JSSProblem(max_numb_of_tasks, max_task_depth, test_set, fixed_max_numbers,
                                    high_numb_of_tasks_preference)
        elif environment == "[J,m=1|pmtn,nowait,tree,nj,t,f,gj=1|avg(T)]":
            env = Jm_tf_T_JSSProblem(max_numb_of_tasks, test_set, fixed_max_numbers, high_numb_of_tasks_preference)
        else:
            env = J_t_D_JSSProblem(max_numb_of_machines, max_numb_of_tasks, max_task_depth, fixed_max_numbers,
                                   high_numb_of_tasks_preference,
                                   high_numb_of_machines_preference, test_set)

        if algorithm == 'supervised':
            dqn_model, history, pretraining_dqn_model = supervised.supervised_learning(env, episodes, batch_size,
                                                                                       get_pretrained_dnn=True)

            # Extracting metrics from the history object as arrays
            training_loss = history.history['loss']
            validation_loss = history.history.get('val_loss', [])  # Empty list if validation loss is not available
            training_accuracy = history.history['accuracy']
            validation_accuracy = history.history.get('val_accuracy',
                                                      [])  # Empty list if validation accuracy is not available
            fitness_curve = training_accuracy

        else:
            # Running the Q-learning algorithm
            dqn_model, fitness_curve, pretraining_dqn_model = dqn.q_learning(env, episodes, gamma, epsilon, alpha,
                                                                             epsilon_decay, min_epsilon,
                                                                             batch_size, update_target_network,
                                                                             get_pretrained_dqn=True,
                                                                             progress_bar=False)

        # Fitness curve calculation and visualization
        vis.show_fitness_curve(fitness_curve, title="Fitness Curve", subtitle=f"DQN")
        print("\n")

        # Environment-specific accuracy computation and visualization
        if environment == "[J,m=1|nowait,f,gj=1|min(T)]":
            print(f"The accuracy of the algorithm is: {validation.time_dqn(test_set, env, dqn_model, less_comments)}%")
            print("This shows the average correctly assorted tasks")
        elif environment == "[J,m=1|pmtn,nowait,tree,nj,t,f,gj=1|avg(T)]":
            print("No validation for this environment exists yet")
        else:
            print(f"The accuracy of the algorithm is: "
                  f"{validation.resource_dqn(test_set, env, dqn_model, less_comments)}/"
                  f"{validation.resource_dqn(test_set, env, pretraining_dqn_model, less_comments)}")
            print("The accuracy is calculated using the Root Mean Squared Error (RMSE). Lower is better.")
            print("The second number is the RMSE of the untrained NN")
        print("")

        if environment != "[J,m=1|pmtn,nowait,tree,nj,t,f,gj=1|avg(T)]":
            # Example execution and visualization
            print("THE EXAMPLE ")
            if environment != "[J,m=1|nowait,f,gj=1|min(T)]":
                print("The Tasks:")
                vis.visualise_tasks(tasks)
            optimal_policy = dqn.get_pi_from_q(env, dqn_model, env.get_specific_state_list([tasks, numb_of_machines]),
                                               less_comments)
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
