import threading
import json
import util


def job_shop(tasks, num_machines):
    num_tasks = len(tasks)
    best_schedule = []
    best_time = float('inf')

    def dfs(schedule, task_index, machine_times):
        nonlocal best_schedule, best_time

        if task_index == num_tasks:
            if max(machine_times) < best_time:
                best_schedule = schedule.copy()
                best_time = max(machine_times)
            return

        for machine in range(num_machines):
            machine_times[machine] += tasks[task_index]
            schedule[task_index] = machine

            if max(machine_times) < best_time:
                dfs(schedule, task_index + 1, machine_times)

            machine_times[machine] -= tasks[task_index]

    initial_schedule = [-1] * num_tasks
    initial_machine_times = [0] * num_machines
    dfs(initial_schedule, 0, initial_machine_times)

    return best_schedule, best_time


def generate_and_solve_job_shop(max_numb_of_machines, max_numb_of_tasks, max_task_depth, high_numb_of_tasks_preference,
                                high_numb_of_machines_preference, fixed_max_numbers):
    # Generate problem instance
    numb_of_machines, numb_of_tasks, tasks = util.generate_specific_job_shop(max_numb_of_machines, max_numb_of_tasks,
                                                                             max_task_depth,
                                                                             high_numb_of_tasks_preference,
                                                                             high_numb_of_machines_preference,
                                                                             fixed_max_numbers)

    # Solve problem instance using job_shop algorithm
    schedule, time = job_shop(tasks, numb_of_machines)

    result_dict = {
        'tasks': tasks,
        'numb_of_machines': int(numb_of_machines),
        'schedule': schedule,
        'time': time
    }
    return result_dict


# Function to run generate_and_solve_job_shop in a separate thread
def run_job_shop_thread(max_numb_of_machines, max_numb_of_tasks, max_task_depth,
                        high_numb_of_tasks_preference, high_numb_of_machines_preference, fixed_max_numbers, results):
    result_dict = generate_and_solve_job_shop(max_numb_of_machines, max_numb_of_tasks, max_task_depth,
                                              high_numb_of_tasks_preference, high_numb_of_machines_preference,
                                              fixed_max_numbers)
    # Append the result to the shared results list
    results.append(result_dict)


def is_unlikely_duplicate(result1, result2):
    tasks1 = result1['tasks']
    tasks2 = result2['tasks']
    num_machines1 = result1['numb_of_machines']
    num_machines2 = result2['numb_of_machines']
    return tasks1 == tasks2 and num_machines1 == num_machines2


def generate_test_data(max_numb_of_machines, max_numb_of_tasks, max_task_depth,
                       high_numb_of_tasks_preference, high_numb_of_machines_preference, numb_of_threads,
                       fixed_max_numbers):
    # Create a list to store the results
    all_results = []

    # Run multiple threads to generate and solve job shop problems
    thread_list = []
    for _ in range(numb_of_threads):
        t = threading.Thread(target=run_job_shop_thread,
                             args=(max_numb_of_machines, max_numb_of_tasks, max_task_depth,
                                   high_numb_of_tasks_preference, high_numb_of_machines_preference, fixed_max_numbers,
                                   all_results))
        thread_list.append(t)
        t.start()

    # Wait for all threads to complete
    for t in thread_list:
        t.join()

    # Remove unlikely duplicates
    filtered_results = []
    for result in all_results:
        is_duplicate = False
        for filtered_result in filtered_results:
            if is_unlikely_duplicate(result, filtered_result):
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_results.append(result)

    # Save the filtered results in a prettier JSON file
    output_data = {
        'results': filtered_results
    }

    # print(output_data)
    output_file = f'test_sets/optimal_job_shobs_mt{max_numb_of_tasks}_mm{max_numb_of_machines}_mtd{max_task_depth}' \
                  f'_len{len(filtered_results)}.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    return output_file, filtered_results
