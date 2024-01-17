import matplotlib.pyplot as plt

# Define a list of colors for representing tasks
task_colors = [
    "#008855", "#880033", "#550088", "#885500", "#007788",
    "#003388", "#338800", "#880077", "#881100", "#778800",
    "#AA5500", "#004477", "#9900AA", "#667700", "#006699",
    "#550099", "#996600", "#997700", "#003344", "#770044"
]

# Unicode character for a square, used for visualization
square_char = "\u25A0"

# Color used for representing machines
machine_grey = "#333333"


def print_hex_color(hex_color, text):
    # Prints text in a specified hex color
    print(f"\033[38;2;{int(hex_color[1:3], 16)};"
          f"{int(hex_color[3:5], 16)};"
          f"{int(hex_color[5:7], 16)}m{text}\033[0m", end='')


def visualise_tasks(tasks):
    # Visualize tasks with colors corresponding to their durations
    i = 1
    for task in tasks:
        if i > 9:
            print(f"Task {chr(ord('A') + i - 10)}: [", end='')
        else:
            print(f"Task {i}: [", end='')
        for j in range(task):
            print_hex_color(task_colors[i], square_char)
        print(f"] ({task})")
        i += 1


def visualise_machines(numb_machines, max_time):
    # Visualize machines with a fixed color for all time points
    for machine in range(numb_machines):
        print(f"Machine {machine}: [", end='')
        for timepoint in range(max_time):
            print_hex_color(machine_grey, square_char)
        print("]")


def visualise_results(optimal_policy, env):
    # Visualize the results of an optimal policy in terms of task allocation on machines
    # The optimal_policy has to be calculated but the result is taken from the env directly
    machines = env.get_result()
    i = 1
    for machine in machines:
        print(f"Machine {i}: [", end='')
        for timepoint in machine:
            if timepoint == 0:
                print_hex_color(machine_grey, square_char)
            else:
                print_hex_color(task_colors[timepoint], square_char)
        print("]  [", end='')
        for timepoint in machine:
            if timepoint == 0:
                print("-", end='')
            else:
                if timepoint > 10:
                    print(chr(ord('A') + timepoint - 10), end='')
                else:
                    print(timepoint, end='')
        print("]")
        i += 1


def show_line_graph(data, legend, title="Fitness Curve", subtitle="", x_label="episodes", y_label="return"):
    # Plot and show a fitness curve using matplotlib
    plt.suptitle(title, fontsize=18)  # title
    plt.title(subtitle, fontsize=10)  # subtitle
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for i in range(len(data)):
        plt.plot(data[i], color=task_colors[i], linewidth=3)
    plt.legend(legend)
    plt.show()


def show_one_line_graph(data, title="Fitness Curve", subtitle="", x_label="episodes", y_label="return"):
    # Plot and show a fitness curve using matplotlib
    plt.suptitle(title, fontsize=18)  # title
    plt.title(subtitle, fontsize=10)  # subtitle
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(data, color="#008855", linewidth=3)
    plt.show()
