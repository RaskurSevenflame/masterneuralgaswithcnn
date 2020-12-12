from self_organizing_maps.Neuron import Neuron
import numpy as np


class NGNeuron(Neuron):
    def calculate_current_neighbourhood_radius(self, start_radius, end_radius, current_iteration, number_of_iterations):
        with_fast_start = False
        exp = False
        if not exp:
            if with_fast_start:
                if current_iteration * 100 < number_of_iterations:
                    return start_radius * ((end_radius / start_radius) ** (current_iteration / number_of_iterations))
                else:
                    return end_radius + ((((np.sqrt(start_radius) * 0.5) ** 2) - end_radius) * (
                                1 - (current_iteration / number_of_iterations)))
            else:
                return end_radius + ((start_radius - end_radius) * (1 - (current_iteration / number_of_iterations)))
        else:
            return start_radius * np.exp(-(current_iteration / number_of_iterations))

    def calculate_distance(self, bmu):
        if type(bmu) is NGNeuron:
            return self.calculate_euclidian_distance_between_neuron_weights(bmu.weights)
        else:
            return self.calculate_euclidian_distance_between_neuron_weights(bmu)
