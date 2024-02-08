import numpy as np
import data_gen_util as dgu

# Random number generators
shuffle = np.random.shuffle
random = np.random.random
randint = np.random.randint
random_choice = np.random.choice

"""
Dict containing the information about the specific data that is used by which environment. 
The numbers represent the position of each attribute in the list

ids = 0
child_foreign_keys = 1
nonpreemtive_flag = 2
lead_time_total = 3
lead_time_todo = 4
processing_time_total = 5
processing_time_todo = 6
deadline (due_date) = 7
done_flag = 8
is_task_ready = 9
"""
env_dict = {
    "[J|nowait,t,gj=1|D]": [5],
    "[J,m=1|nowait,f,gj=1|T]": [7],
    "[J,m=1|pmtn,nowait,tree,nj,t,f,gj=1|T]": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}


def weighted_randint(max_numb, high_value_preference):
    # Returns a weighted random integer, with higher preference for larger numbers
    i = np.arange(max_numb + 1)
    adjusted_preference = 4 * high_value_preference
    weights = np.power(i, adjusted_preference)
    weights /= weights.sum()
    numb = np.random.choice(i, p=weights)
    return numb


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


# execute this method to generate a new dataset
def generate_new_dataset(episodes: int, numb_of_tasks: int):
    for episode in range(episodes):
        ids = np.arange(0, numb_of_tasks)
        child_foreign_keys = dgu.validate_child_elements(numb_of_tasks)
        nonpreemtive_flag = random_choice(np.arange(0, 2), size=numb_of_tasks)
        lead_time_total = random_choice(np.arange(0, 5), size=numb_of_tasks, p=[1 / 2] + [1 / 8] * 4)
        lead_time_todo = lead_time_total
        processing_time_total = random_choice(np.arange(1, 10), size=numb_of_tasks)
        processing_time_todo = processing_time_total
        deadline = dgu.generate_deadlines_with_target_average(numb_of_tasks, 40, (10, 50))
        done_flag = np.zeros(numb_of_tasks, dtype=int)
        is_task_ready = np.ones(numb_of_tasks, dtype=int)

        result_list = list(
            [ids, child_foreign_keys, nonpreemtive_flag, lead_time_total, lead_time_todo, processing_time_total,
             processing_time_todo, deadline, done_flag, is_task_ready])

        dgu.save_training_data_as_pkl_file(str(episode), result_list, episodes, numb_of_tasks)


def get_start_state(env_name: str, number_of_tasks: int, num_episode: int, dir_name: str):
    result_list = dgu.read_list_from_pkl_file(str(num_episode), dir_name)
    temp_result_list = []
    for item in env_dict.get(env_name):
        temp_result_list.append(result_list[item][:number_of_tasks])
    if 1 in env_dict.get(env_name):
        temp_result_list = dgu.remove_child_element_keys_that_are_too_high(temp_result_list)
    return temp_result_list


def label_training_data(env, epochs: int, number_of_tasks: int, env_name: str, unlabeled_data_dir_name: str):
    dataset = create_correct_histories(env, epochs, result_as_unsorted_state_action_pairs=True)

    dgu.save_labeled_data_as_pkl_file(dataset, epochs, number_of_tasks, env_name, unlabeled_data_dir_name)


def create_correct_histories(env, epochs: int, result_as_unsorted_state_action_pairs: bool=False):
    dataset = list()
    print(epochs)
    # TODO: Why only 282 datapoints if 1000 requested
    for epoch in range(epochs):
        state = env.get_start_state(epoch)
        data_item = list()

        while not env.done(state):

            # Get the correct action
            action = env.get_correct_action(state)
            if not action:
                break
            action_index = env.action_to_int(action)

            # Take action, observe next state and correctness
            next_state = env.get_next_state(state, action)

            # Save data to dataset
            data_item.append({"state": env.pad_state(state), "action": action_index})

            # Update state
            state = list(next_state)

        if result_as_unsorted_state_action_pairs:
            dataset.extend(data_item)
        else:
            dataset.append(data_item)

    if result_as_unsorted_state_action_pairs:
        shuffle(dataset)

    print(len(dataset))

    return dataset
