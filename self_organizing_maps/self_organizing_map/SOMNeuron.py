from self_organizing_maps.Neuron import Neuron
import numpy as np


class SOMNeuron(Neuron):
    def calculate_current_neighbourhood_radius(self, start_radius, end_radius, current_iteration, number_of_iterations):
        exp = False
        if not exp:
            return end_radius + ((start_radius - end_radius) * (1 - (current_iteration / number_of_iterations)))
        else:
            return start_radius * np.exp(-(current_iteration / number_of_iterations))

    def calculate_distance(self, best_matching_unit):
        if type(best_matching_unit) is SOMNeuron:
            return np.sqrt(((self.x_center - best_matching_unit.x_center) * (self.x_center - best_matching_unit.x_center)) + (
                    (self.y_center - best_matching_unit.y_center) * (self.y_center - best_matching_unit.y_center)))
        else:
            return self.calculate_euclidian_distance_between_neuron_weights(best_matching_unit)