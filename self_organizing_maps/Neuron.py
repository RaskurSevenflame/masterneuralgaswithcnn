import numpy as np
import abc


class Neuron(object):
    def __init__(self, dimensions, x_axis_count,  y_axis_count, x_axis_length, y_axis_length, map_radius, algorithm,
                 number_of_iterations):
        self.algorithm = algorithm

        # initialized weights randomly between 0 and 1 as the data is normalized
        self.weights = np.random.rand(dimensions, 1)
        self.dimensions = dimensions

        # place in the grid
        self.x_axis_counter = x_axis_count
        self.y_axis_counter = y_axis_count

        # saves the center of your neuron (this is only constant in SOM)
        self.x_center = (self.x_axis_counter + 0.5) / x_axis_length
        self.y_center = (self.y_axis_counter + 0.5) / y_axis_length

        self.number_of_iterations = number_of_iterations
        self.map_radius = map_radius

    def adjust_axis_values(self, x_axis_value, y_axis_value, x_axis_length, y_axis_length):
        self.x_axis_counter = x_axis_value
        self.y_axis_counter = y_axis_value
        self.x_center = (self.x_axis_counter + 0.5) / x_axis_length
        self.y_center = (self.y_axis_counter + 0.5) / y_axis_length

    @abc.abstractmethod
    def calculate_distance(self, bmu):
        pass

    def set_weights(self, vector):
        self.weights = vector

    def adjust_weights(self, learning_rate, influence, input_vector):
        alpha = learning_rate
        if learning_rate > 1:
            alpha = 1
        beta = round(alpha * influence, 10)

        self.weights = self.weights + beta * (input_vector - self.weights)

    @abc.abstractmethod
    def calculate_current_neighbourhood_radius(self, start_radius, end_radius, current_iteration, number_of_iterations):
        pass

    def calculate_influence(self, distance, radius):
        # the following line is a recommendation from wiki
        # return np.exp(-(distance * distance) / (2 * radius * radius))
        return np.exp(-( distance ) / ( radius ))

    def calculate_euclidian_distance_between_neuron_weights(self, bmu_weights):
        return np.linalg.norm(self.weights - bmu_weights)
