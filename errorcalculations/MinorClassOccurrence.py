import numpy as np
from errorcalculations.MeasureErrorValue import MeasureErrorValue
import math
from errorcalculations.CrossEntropy import CrossEntropy


class MinorClassOccurrence(MeasureErrorValue):

    """
        Measures the occurence of minor-classes in the neurons
    """

    def measure_error(self, data, neurons, base, label, amount_of_different_labels, after_dimension_reduction):
        global_error = 0
        grid = self.generate_label_counting_grid(neurons, data, label, base, amount_of_different_labels)

        for row in grid:
            for col in row:
                global_error += np.sum(col) - np.max(col)


        return global_error / len(data)

    def get_name(self):
        return "MinorClassOccurrence"
