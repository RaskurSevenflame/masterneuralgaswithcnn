import abc
import math
import numpy as np

class Sorter(object):

    """
        base-class of a Self-Organzing Maps algorithm
    """

    @abc.abstractmethod
    def calculate_number_of_neurons(self, x_axis_length, y_axis_length) -> int:
        pass

    @abc.abstractmethod
    def find_neighbourhood(self, neurons, best_matching_unit, radius, vector) -> []:
        pass

    @abc.abstractmethod
    def find_x_nearest_neighbors(self, neurons, best_matching_unit, radius, vector, fixed_neighborhood_size) -> []:
        pass

    @abc.abstractmethod
    def get_sorted_weights(self, x_axis_length, y_axis_length, neurons) -> []:
        pass

    @abc.abstractmethod
    def calculate_learningrate(self, start_learning_rate, end_learningrate, current_iteration, number_of_iterations):
        pass

    @abc.abstractmethod
    def calculate_starting_radius(self, dimensions, x_axis, y_axis, start_radius_multiplikator):
        pass

    @abc.abstractmethod
    def calculate_end_radius(self, dimensions, x_axis, y_axis, start_radius_multiplicator, start_radius):
        pass

    @abc.abstractmethod
    def get_name(self):
        pass

    @staticmethod
    def find_direct_neighbour(neurons, current_unit_weights, already_used):
        neighbour = None

        smallest_distance = math.inf
        for n in neurons:
            if n not in already_used:
                distance = np.linalg.norm(n.weights - current_unit_weights)
                if smallest_distance > distance:
                    smallest_distance = distance
                    neighbour = n

        return neighbour

