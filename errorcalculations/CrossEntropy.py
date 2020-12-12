import numpy as np
from errorcalculations.MeasureErrorValue import MeasureErrorValue
import math


class CrossEntropy(MeasureErrorValue):

    """
        Calculates the main-class of each neuron an measures the amount regarding to minor-classes of the same neuron
    """

    def measure_error(self, data, neurons, base, label, amount_of_different_labels, after_dimension_reduction):
        global_error = 0

        grid = self.generate_label_counting_grid(neurons, data, label, base, amount_of_different_labels)

        for row in grid:
            for col in row:
                if np.sum(col) > 0:
                    global_error += -1 * np.log(np.max(col) / np.sum(col))
                else:
                    print("sum of col is zero; which would lead to dividing by zero; in CrossEntropy")

        return global_error

    def get_name(self):
        return "CrossEntropy"
