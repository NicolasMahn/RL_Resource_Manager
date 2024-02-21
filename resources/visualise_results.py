import matplotlib.pyplot as plt
import numpy as np

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


def show_line_graph(data, legend, title="Fitness Curve", subtitle="", x_label="episodes", y_label="return",
                    start_with_zero: bool = False):
    # Plot and show a fitness curve using matplotlib
    plt.suptitle(title, fontsize=18)  # title
    plt.title(subtitle, fontsize=10)  # subtitle
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for i in range(len(data)):
        if start_with_zero:
            plt.plot(([0] + data[i]), color=task_colors[i], linewidth=3)
        else:
            plt.plot(data[i], color=task_colors[i], linewidth=3)
    plt.legend(legend)
    plt.show()


def show_one_line_graph(data, title="Fitness Curve", subtitle="", x_label="episodes", y_label="return",
                        start_with_zero: bool = False):
    # Plot and show a fitness curve using matplotlib
    fig, ax = plt.subplots()
    plt.suptitle(title, fontsize=18)  # title
    plt.title(subtitle, fontsize=10)  # subtitle
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if start_with_zero:
        plt.plot(([0] + data), color="#008855", linewidth=3)
    else:
        plt.plot(data, color="#008855", linewidth=3)
    plt.show()
    
    return fig


def get_polynomial_fitness_curve(fitness_curve, degree):
    # Perform polynomial regression
    x = np.arange(len(fitness_curve))
    coefficients = np.polyfit(x, fitness_curve, degree)
    polynomial = np.poly1d(coefficients)

    # Generate x values for plotting the fitted polynomial curve
    x_poly = np.linspace(x[0], x[-1], len(fitness_curve))
    return polynomial(x_poly)
