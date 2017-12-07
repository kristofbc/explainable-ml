import numpy

"""
    Count how many time a value is found inside the data
    Args:
        values (mixed): The values to count
        value_function (lambda): A function that returns the value to count
    Returns:
        dictionary
"""
def count_unique_value(values, value_function=lambda values, cur_index: values[cur_index]):
    unique = {}
    for i in range(len(values)):
        value = value_function(values, i)
        if value not in unique:
            unique[value] = 0.0
        unique[value] += 1.0
    return unique
