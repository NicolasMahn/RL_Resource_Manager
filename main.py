import numpy as np
import util
from environments.resource_management import ResourceManagement
import algorithms.dqn as alg
import matplotlib.pyplot as plt
import visualise_results as vis
import time


def main():

    tasks = [4, 1, 2]
    numb_of_machines = 2

    print("The Tasks:")
    vis.visualise_tasks(tasks)

    print("\nThe Machines:")
    vis.visualise_machines(numb_of_machines, sum(tasks))
    time.sleep(0.1)

    print("\nTraining the DQN model...")
    env = ResourceManagement(numb_of_machines=numb_of_machines, tasks=tasks)
    dqn_model, fitness_curve = alg.q_learning(env, episodes=400, updates=True)
    show_fitness_curve(fitness_curve, subtitle="DQN")
    print("")

    # Obtain the optimal policy
    optimal_policy = alg.get_pi_from_q(env, dqn_model)

    print("The optimal policy:")
    vis.visualise_results(optimal_policy, env)


def show_fitness_curve(data, title="Fitness Curve", subtitle="", x_label="episodes", y_label="return"):
    plt.suptitle(title, fontsize=18)  # title
    plt.title(subtitle, fontsize=10)  # subtitle
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot([data[i] for i in range(0, len(data))], color="#008855", linewidth=3)
    plt.show()


if __name__ == '__main__':
    main()

