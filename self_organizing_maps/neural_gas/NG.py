from self_organizing_maps.Sorter import Sorter
import numpy as np
import math
from self_organizing_maps.neural_gas.NGNeuron import NGNeuron


class NG(Sorter):

    def get_name(self):
        return "Neural_Gas"

    def calculate_number_of_neurons(self, x_axis_length, y_axis_length) -> int:
        return x_axis_length * y_axis_length

    def find_neighbourhood(self, neurons, best_matching_unit, radius, vector):
        neighbourhood = []
        for n in neurons:
            distance = np.linalg.norm(n.weights - best_matching_unit.weights)
            if distance <= radius:
                neighbourhood.append(n)
        return neighbourhood, neurons

    def find_x_nearest_neighbors(self, neurons, best_matching_unit, radius, vector, fixed_neighborhood_size):
        # instead of calculating the neurons only in a radius, the amount of neurons is limited to x = fixed_neighborhood_size
        neighbourhood = []
        already_used_neurons = []
        for i in range(fixed_neighborhood_size):
            distance = math.inf
            neuron_number = 0
            neighbour = None
            none_found = True
            for j in range(len(neurons)):
                if j not in already_used_neurons:
                    if distance > neurons[j].calculate_distance(best_matching_unit) < radius:
                        distance = neurons[j].calculate_distance(best_matching_unit)
                        neighbour = neurons[j]
                        neuron_number = j
                        none_found = False
            if none_found:
                break
            neighbourhood.append(neighbour)
            already_used_neurons.append(neuron_number)

        return neighbourhood, neurons

    def get_sorted_weights(self, x_axis_length, y_axis_length, neurons) -> []:
        # as the neurons needed to be sorted for some of the early grid creations
        weight_array = []
        already_used = []
        current_neuron = neurons[0]
        weight_array.append(current_neuron.weights.reshape(1, len(neurons[0].weights))[0])
        already_used.append(current_neuron)
        count = 1
        while len(neurons) != count:
            current_neuron = self.find_direct_neighbour(neurons, current_neuron.weights, already_used)
            weight_array.append(current_neuron.weights.reshape(1, len(neurons[0].weights))[0])
            already_used.append(current_neuron)
            count = count + 1

        return weight_array

    def create_neuron(self, dimensions, algorithm, number_of_iterations, x_axis_counter, y_axis_counter, grid_radius,
                      x_axis_length, y_axis_length):
        neuron = NGNeuron(dimensions, x_axis_counter, y_axis_counter, x_axis_length, y_axis_length, grid_radius,
                          algorithm, number_of_iterations)
        return neuron

    def calculate_learningrate(self, start_learning_rate, end_learningrate, current_iteration, number_of_iterations):
        return end_learningrate + (
                (start_learning_rate - end_learningrate) * (1 - (current_iteration / number_of_iterations)))

    def calculate_starting_radius(self, dimensions, x_axis, y_axis, start_radius_multiplicator):
        return np.sqrt(dimensions) * start_radius_multiplicator

    def calculate_end_radius(self, dimensions, x_axis, y_axis, start_radius_multiplicator, start_radius):
        return start_radius / np.sqrt(dimensions)

    @staticmethod
    def calculate_dummy_two_dimensional_neighbouthood_amount(number_of_neurons, current_iteration,
                                                             number_of_iterations):
        number = int(number_of_neurons * 0.5 * ((1 - (current_iteration / number_of_iterations)) ** 2))
        if number < 1:
            number = 1
        return number
