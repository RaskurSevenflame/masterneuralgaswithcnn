from self_organizing_maps.Sorter import Sorter
import math
from self_organizing_maps.self_organizing_map.SOMNeuron import SOMNeuron

class SOM(Sorter):

    def get_name(self):
        return "Self-Organizing_Map"

    def calculate_number_of_neurons(self, x_axis_length, y_axis_length) -> int:
        return x_axis_length * y_axis_length

    def find_neighbourhood(self, neurons, best_matching_unit, radius, vector):
        neighbourhood = []

        for neuron in neurons:
            distance = neuron.calculate_distance(best_matching_unit)
            if distance <= radius:
                neighbourhood.append(neuron)

        return neighbourhood, neurons

    def find_x_nearest_neighbors(self, neurons, best_matching_unit, radius, vector, fixed_neighborhood_size):
        neighbourhood = []
        not_used_neurons = neurons
        for i in range(fixed_neighborhood_size):
            distance = math.inf
            neighbour = None
            for neuron in not_used_neurons:
                if distance > neuron.calculate_distance(best_matching_unit):
                    distance = neuron.calculate_distance(best_matching_unit)
                    neighbour = neuron
            neighbourhood.append(neighbour)
            not_used_neurons.remove(neighbour)

        return neighbourhood, neurons

    def get_sorted_weights(self, x_axis_length, y_axis_length, neurons) -> []:
        weight_array = []
        for i in range(y_axis_length):
            for j in range(x_axis_length):
                weight_array.append(list(
                    neurons[j + (i * x_axis_length)].weights.reshape(1, len(neurons[j + (i * x_axis_length)].weights))[
                        0]))

        return weight_array

    def create_neuron(self, dimensions, algorithm, number_of_iterations, x_axis_counter, y_axis_counter, grid_radius, x_axis_length, y_axis_length):
        neuron = SOMNeuron(dimensions, x_axis_counter, y_axis_counter, x_axis_length, y_axis_length, grid_radius,
                           algorithm, number_of_iterations)

        return neuron

    def calculate_learningrate(self, start_learning_rate, end_learningrate, current_iteration, number_of_iterations):
        learningrate = end_learningrate + ((start_learning_rate - end_learningrate) * (1 - (current_iteration / number_of_iterations)))
        return learningrate

    def calculate_starting_radius(self, dimensions, x_axis, y_axis, start_radius_multiplicator):
        return start_radius_multiplicator

    def calculate_end_radius(self, dimensions, x_axis, y_axis, start_radius_multiplicator, start_radius):
        return 0.5 / max(x_axis, y_axis)
