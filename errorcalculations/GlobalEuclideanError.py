import numpy as np
from errorcalculations.MeasureErrorValue import MeasureErrorValue


class GlobalEuclideanError(MeasureErrorValue):

    """
        Calculates the difference in distance of each neuron to the data-vectors
    """

    def measure_error(self, data, neurons, base, label, amount_of_different_labels, after_dimension_reduction):
        global_error = 0
        dimensions = len(data[0])
        for vector in data:
            input_vector = np.asarray(vector)
            reshaped_vector = input_vector.reshape(dimensions, 1)
            bmu = base.find_best_matching_unit(reshaped_vector, neurons)
            global_error += bmu.calculate_euclidian_distance_between_neuron_weights(reshaped_vector)

        global_error = global_error / np.sqrt(len(data[0]))
        return global_error

    def get_name(self):
        return "GlobalEuclideanError"
