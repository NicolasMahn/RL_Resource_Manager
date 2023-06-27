import numpy as np
import math

random = np.random.random
randint = np.random.randint


# numpys argmax just takes the first result if there are several
# that is problematic and the reason for a new argmax func
def argmax(elements):
    # print(f"elements: {elements}")
    best_elements = argmax_multi(elements)
    return best_elements[randint(0, len(best_elements))]


def argmax_multi(elements):
    max_element = np.max(elements)
    # print(f"max_element: {max_element}")
    if math.isnan(max_element):
        raise ValueError("NaN in argmax_multi")
    best_elements = [i for i in range(len(elements)) if elements[i] == max_element]
    return best_elements


def flatmap_list(lst):
    result = []
    for item in lst:
        if isinstance(item, list) or isinstance(item, np.ndarray):
            result.extend(flatmap_list(item))
        else:
            result.append(item)
    return result


def binary_to_decimal(binary_list):
    decimal = 0
    for bit in binary_list:
        decimal = decimal * 2 + bit
    return decimal


def decimal_to_binary(decimal_number, length=None):
    binary_list = []
    while decimal_number > 0:
        bit = decimal_number % 2
        binary_list.insert(0, bit)
        decimal_number //= 2

    # Pad the binary list with leading zeros if a length is specified
    if length is not None:
        while len(binary_list) < length:
            binary_list.insert(0, 0)

    return binary_list

