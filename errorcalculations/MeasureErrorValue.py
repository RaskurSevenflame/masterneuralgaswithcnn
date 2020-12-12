import abc
import math
import numpy as np


class MeasureErrorValue:

    """
        Abstract class for each error-calculation
    """

    @abc.abstractmethod
    def measure_error(self, data, neurons, base, label, amount_of_different_labels, after_dimension_reduction):
        pass

    @abc.abstractmethod
    def get_name(self):
        pass

    @staticmethod
    def generate_label_counting_grid(neurons, data, label, base, amount_of_different_labels):

        grid = np.zeros([base.y_axis_length, base.x_axis_length, amount_of_different_labels])
        grid = np.asarray(grid)
        if len(grid) * len(grid[0]) != len(neurons):
            print("GNG has more Neurons than Gridtiles")
            return None

        for i in range(len(data)):
            input_vector = data[i]
            input = input_vector.reshape(base.dimensions, 1)
            bmu = base.find_best_matching_unit(input, neurons)
            grid[bmu.y_axis_counter][bmu.x_axis_counter][int(label[i])] += 1

        return grid
