from self_organizing_maps.Sorter import Sorter
from self_organizing_maps.growing_neural_gas.GNGNeuron import GNGNeuron
import math
import numpy as np


class GNG(Sorter):
    """
        this variation of the normal growing neural gas algorithm is explained in my thesis.
        in short the oder of some executions is changed, as those changes have almost no impact on the result,
        but ensure that the algrithm can be used in the basis-algorithm
    """

    def get_name(self):
        return "Growing_Neural_Gas"

    def __init__(self):
        # this parameters could be optimized
        self.number_of_iterations_until_a_neuron_is_created = 100
        self.age_maximum = 7
        self.max_unused_age = 350
        self.infinite_grow_allowed = False
        self.decreasing_learningrate = False

        # needed to generate the first connection
        self.first_neuron = None
        # saves all connections between neurons
        self.connections = []

        self.current_number_of_neurons = 0
        self.total_number_of_neurons_created = 0
        self.max_number_of_neurons = math.inf

    def calculate_number_of_neurons(self, x_axis_length, y_axis_length) -> int:
        if self.infinite_grow_allowed:
            self.max_number_of_neurons = x_axis_length * y_axis_length
        return 2  # as the startingnumber of neurons is two

    def calculate_error_for_a_neuron(self, bmu, vector):
        bmu.error_value += bmu.calculate_distance(vector)

    def find_x_nearest_neighbors(self, neurons, best_matching_unit, radius, vector, fixed_neighborhood_size):
        print("Finding only x neurons in the neighborhood does not work for GNG")
        return self.find_neighbourhood(neurons, best_matching_unit, radius, vector)

    def find_neighbourhood(self, neurons, best_matching_unit, radius, vector):
        neighbor = self.find_direct_neighbour(neurons, vector, [best_matching_unit])
        connection_exists = False
        for connection in self.connections:
            if (connection[0].number == best_matching_unit.number and connection[1].number == neighbor.number) or (
                    connection[1].number == best_matching_unit.number and connection[0].number == neighbor.number):
                connection_exists = True
                connection[2] = 0
                # this age is presented in my thesis and tries to deleted unused neurons
                # that would remain between clusters otherwise
                connection[3] = 0

        if not connection_exists:
            self.connections.append([best_matching_unit, neighbor, 0, 0])

        self.calculate_error_for_a_neuron(best_matching_unit, vector)
        self.increase_age_of_emanating_connections(best_matching_unit)
        self.increase_second_age_of_all_connections()
        neurons = self.remove_old_connections(neurons)

        neighbors = []
        for connection in self.connections:
            if connection[0].number == best_matching_unit.number:
                neighbors.append(connection[1])
            if connection[1].number == best_matching_unit.number:
                neighbors.append(connection[0])

        return neighbors, neurons

    def increase_second_age_of_all_connections(self):
        for connection in self.connections:
            connection[3] += 1

    def remove_old_connections(self, neurons):
        removing = False
        for connection in self.connections:
            if connection[2] > self.age_maximum or connection[3] > self.max_unused_age:
                self.connections.remove(connection)
                removing = True

        neurons_to_remove = []
        for neuron in neurons:
            has_no_connection = True
            for connection in self.connections:
                if connection[0].number == neuron.number or connection[1].number == neuron.number:
                    has_no_connection = False
            if has_no_connection:
                neurons_to_remove.append(neuron)

        for neuron in neurons_to_remove:
            neurons.remove(neuron)
            self.current_number_of_neurons = self.current_number_of_neurons - 1

        return neurons

    def create_neuron(self, dimensions, algorithm, number_of_iterations, x_axis_counter,
                      y_axis_counter, grid_radius, x_axis_length, y_axis_length):

        neuron = GNGNeuron(dimensions, x_axis_counter, y_axis_counter, x_axis_length, y_axis_length, grid_radius,
                           algorithm, number_of_iterations)

        neuron.number = self.total_number_of_neurons_created
        self.total_number_of_neurons_created += 1
        self.current_number_of_neurons += 1
        if self.total_number_of_neurons_created == 1:
            self.first_neuron = neuron
        if self.total_number_of_neurons_created == 2:
            self.connections.append([self.first_neuron, neuron, 0, 0])
        return neuron

    def increase_age_of_emanating_connections(self, neuron):
        for connection in self.connections:
            if connection[0].number == neuron.number or connection[1].number == neuron.number:
                connection[2] = connection[2] + 1

    def calculate_learningrate(self, start_learning_rate, end_learningrate, current_iteration, number_of_iterations):
        if self.decreasing_learningrate:
            return min([start_learning_rate * (1 - (current_iteration / number_of_iterations)), end_learningrate])
        else:
            return start_learning_rate

    def insert_new_neuron(self, error_reduction, neurons, dimensions, algorithm, number_of_iterations, grid_radius,
                          x_axis_length, y_axis_length):
        highest_error_neuron = self.get_highest_error_neuron(neurons)

        neighbor_neuron_with_highest_error = None
        neighbor_highest_error = -1
        connection_to_remove = None
        for connection in self.connections:
            if connection[0].number == highest_error_neuron.number and connection[
                1].error_value > neighbor_highest_error:
                neighbor_highest_error = connection[1].error_value
                neighbor_neuron_with_highest_error = connection[1]
                connection_to_remove = connection
            if connection[1].number == highest_error_neuron.number and connection[
                0].error_value > neighbor_highest_error:
                neighbor_highest_error = connection[0].error_value
                neighbor_neuron_with_highest_error = connection[0]
                connection_to_remove = connection

        new_neuron_vector = 0.5 * (neighbor_neuron_with_highest_error.weights + highest_error_neuron.weights)
        neuron = self.create_neuron(dimensions, algorithm, number_of_iterations, 0, 0, grid_radius, x_axis_length,
                                    y_axis_length)
        neuron.set_weights(new_neuron_vector)
        neurons.append(neuron)

        self.connections.remove(connection_to_remove)
        self.connections.append([highest_error_neuron, neuron, 0, 0])
        self.connections.append([neuron, neighbor_neuron_with_highest_error, 0, 0])

        # there is no reason to use learningrate here, there could be usage of another value like the radius
        highest_error_neuron.error_value *= error_reduction
        neighbor_neuron_with_highest_error.error_value *= error_reduction
        neuron.error_value = highest_error_neuron.error_value

        return neurons

    def post_weight_adjustment(self, neurons, learning_rate, error_reduction_for_new_neuron, current_iteration,
                               dimensions, algorithm, number_of_iterations, grid_radius, x_axis_length, y_axis_length):
        if current_iteration % self.number_of_iterations_until_a_neuron_is_created == 0 and self.current_number_of_neurons < self.max_number_of_neurons:
            neurons = self.insert_new_neuron(error_reduction_for_new_neuron, neurons, dimensions, algorithm,
                                             number_of_iterations, grid_radius, x_axis_length, y_axis_length)

        neurons = self.reduce_error_values(neurons, learning_rate)
        return neurons

    def reduce_error_values(self, neurons, learning_rate):
        for neuron in neurons:
            neuron.error_value = neuron.error_value * learning_rate
            neuron.error_value = np.round(neuron.error_value, 7)
        return neurons

    def calculate_starting_radius(self, dimensions, x_axis, y_axis, start_radius_multiplikator):
        return start_radius_multiplikator

    def calculate_end_radius(self, dimensions, x_axis, y_axis, start_radius_multiplicator, start_radius):
        return 0

    def fix_number_of_neurons(self, neurons, number_to_fix_to, x_axis_length, y_axis_length,
                              error_reduction_for_new_neuron, algorithm, dimensions):
        while len(neurons) != number_to_fix_to:
            if len(neurons) < number_to_fix_to:
                neurons = self.insert_new_neuron(error_reduction_for_new_neuron, neurons, dimensions, algorithm, 0, 0,
                                                 x_axis_length, y_axis_length)
            else:
                neurons = self.remove_neuron_with_highest_error_value(neurons)

        return neurons

    def remove_neuron_with_highest_error_value(self, neurons):
        highest_error_neuron = self.get_highest_error_neuron(neurons)
        neurons.remove(highest_error_neuron)
        return neurons

    def get_highest_error_neuron(self, neurons):
        highest_error = -1
        highest_error_neuron = None
        for neuron in neurons:
            if highest_error < neuron.error_value:
                highest_error = neuron.error_value
                highest_error_neuron = neuron
        return highest_error_neuron
