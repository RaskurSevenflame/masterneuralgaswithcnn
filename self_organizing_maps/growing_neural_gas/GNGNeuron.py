from self_organizing_maps.Neuron import Neuron
import numpy as np


class GNGNeuron(Neuron):

    def __init__(self, dimensions, x_axis_counter, x_axis_max,
                 y_axis_counter, y_axis_max, grid_radius,
                 algorithm, number_of_iterations):
        super(GNGNeuron, self).__init__(dimensions, x_axis_counter, x_axis_max,
                                        y_axis_counter, y_axis_max, grid_radius,
                                        algorithm, number_of_iterations)

        self.error_value = 0.0
        self.number = None

    def calculate_current_neighbourhood_radius(self, start_radius, end_radius, current_iteration, number_of_iterations):
        # the radius is not needed for the gng
        return start_radius

    def calculate_influence(self, distance, radius):
        return np.exp(-(distance * distance) / (2 * radius * radius))

    def calculate_distance(self, bmu):
        if type(bmu) is GNGNeuron:
            return self.calculate_euclidian_distance_between_neuron_weights(bmu.weights)
        else:
            return self.calculate_euclidian_distance_between_neuron_weights(bmu)
