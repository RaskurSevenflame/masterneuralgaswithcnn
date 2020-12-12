import numpy as np
import random
from self_organizing_maps.growing_neural_gas.GNG import GNG
from self_organizing_maps.self_organizing_map.SOM import SOM
from self_organizing_maps.neural_gas.NG import NG


class Base(object):
    def __init__(self, grid_x_axis_length, grid_y_axis_length, algorithm, data,
                 number_of_iterations, start_learning_rate, start_radius_multiplicator,
                 end_learning_rate, random_type):

        # initialize all needed parameters, which are generated from the currently used algorithm

        # determine which sorter is used
        self.algorithm = self.create_new_algorithm(algorithm)

        self.data = data
        self.start_learning_rate = start_learning_rate
        self.learning_rate = self.start_learning_rate

        self.number_of_iterations = number_of_iterations

        self.x_axis_length = grid_x_axis_length
        self.y_axis_length = grid_y_axis_length

        self.random_type = random_type

        if len(self.data) == 0 or len(self.data[0]) == 0:
            self.error = "len(data) == 0"
        self.dimensions = len(self.data[0])
        self.number_of_neurons = self.algorithm.calculate_number_of_neurons(grid_x_axis_length, grid_y_axis_length)

        self.end_learning_rate = end_learning_rate
        self.start_radius = self.algorithm.calculate_starting_radius(self.dimensions, grid_x_axis_length,
                                                                     grid_y_axis_length,
                                                                     start_radius_multiplicator)
        self.end_radius = self.algorithm.calculate_end_radius(self.dimensions, self.x_axis_length, self.y_axis_length,
                                                              start_radius_multiplicator, self.start_radius)

        self.createRandomInputs()

        # Creates them with number of neurons being the columns
        self.neurons = self.create_neurons(self.algorithm, self.dimensions, self.number_of_iterations,
                                           self.start_radius, self.number_of_neurons)

    def train(self):
        """
            this algorithm is explained in my thesis, it ensures the comparison between the algorithms
            as it executes the same code for all of them, only using the specific code of an algorithm
            if the differ from each other.
            In short a vector from the dataset is presented to the neurons,
            which calculate the neuron that is closest to it, as the (best-matching-unit (bmu)
            from the bmu the radius is drawn calculating all neighbouring neurons
            all neurons inside the radius are then moved to the vector by a fraction defined by the learningrate & distane
            finally the learningrate is updated
            this iterates until the iterationlimit is reached
        """
        for current_iteration in range(self.number_of_iterations):
            input_vector = self.get_random_input()

            best_matching_unit = self.find_best_matching_unit(input_vector, self.neurons)
            radius = best_matching_unit.calculate_current_neighbourhood_radius(self.start_radius, self.end_radius,
                                                                               current_iteration,
                                                                               self.number_of_iterations)

            neighbourhood = self.find_neighbourhood(current_iteration, best_matching_unit, radius, input_vector)

            # Node: the best_matching_unit is inside the neigbourhood-array
            for neighbour in neighbourhood:
                distance = neighbour.calculate_distance(best_matching_unit)
                influence = neighbour.calculate_influence(distance, radius)
                neighbour.adjust_weights(self.learning_rate, influence, input_vector)

            self.learning_rate = self.algorithm.calculate_learningrate(self.start_learning_rate, self.end_learning_rate,
                                                                       current_iteration, self.number_of_iterations)
            self.learning_rate = round(self.learning_rate, 7)
            if type(self.algorithm) is GNG:
                self.neurons = self.algorithm.post_weight_adjustment(self.neurons, self.learning_rate, radius,
                                                                     current_iteration, self.dimensions, self.algorithm,
                                                                     self.number_of_iterations, self.start_radius,
                                                                     self.x_axis_length, self.y_axis_length)

        if type(self.algorithm) is GNG and len(self.neurons) != self.x_axis_length * self.y_axis_length:
            self.neurons = self.algorithm.fix_number_of_neurons(self.neurons, self.x_axis_length * self.y_axis_length,
                                                                self.x_axis_length, self.y_axis_length, 0.1,
                                                                self.algorithm, self.dimensions)
        return self.neurons

    def get_neurons(self):
        return self.neurons

    def get_random_input(self):
        radom_data_vector = 0
        if self.random_type == "cicle":
            if len(self.current_inputs) > 0:
                random_value = random.randint(0, len(self.current_inputs) - 1)
                input_vector = np.asarray(self.current_inputs[random_value])
                self.current_inputs.pop(random_value)
                radom_data_vector = input_vector.reshape(self.dimensions, 1)
            elif len(self.current_inputs) == 0 and len(self.data) > 0:
                self.createRandomInputs()
                return self.get_random_input()
            else:
                print("An Error occured in getRandomInput('cicle')")
        elif self.random_type == "not_random":
            if len(self.current_inputs) > 0:
                input_vector = np.asarray(self.current_inputs[len(self.current_inputs) - 1])
                self.current_inputs.pop(len(self.current_inputs) - 1)
                radom_data_vector = input_vector.reshape(self.dimensions, 1)
            elif len(self.current_inputs) == 0 and len(self.data) > 0:
                self.createRandomInputs()
                return self.get_random_input()
            else:
                print("An Error occured in getRandomInput('not_random')")
        else:  # default
            random_value = random.randint(0, len(self.data) - 1)
            input_vector = np.asarray(self.data[random_value])
            radom_data_vector = input_vector.reshape(self.dimensions, 1)
        return radom_data_vector

    def find_best_matching_unit(self, input_vector, neurons):
        # find the closest unit to a given vector
        best_matching_unit = neurons[0]
        distance_of_the_best_matching_unit = neurons[0].calculate_distance(input_vector)
        for neuron in neurons:
            distance = neuron.calculate_distance(input_vector)
            if distance < distance_of_the_best_matching_unit:
                best_matching_unit = neuron
                distance_of_the_best_matching_unit = distance

        return best_matching_unit

    def createRandomInputs(self):
        self.current_inputs = []
        for point in self.data:
            self.current_inputs.append(point)

    def predict(self, element):
        return self.find_best_matching_unit(element, self.neurons)

    def generate_neurons_from_neuron_information(self, neuron_informations, algorithm, dimensions, number_of_iterations,
                                                 start_radius):
        # with neuron-information from a pickle-file the neurons can be recreated, which is done here
        neurons = self.create_neurons(algorithm, dimensions, number_of_iterations, start_radius,
                                      len(neuron_informations))
        for i in range(len(neuron_informations)):
            weight_vector = neuron_informations[i][1]
            neurons[i].set_weights(weight_vector)
            axis_values = neuron_informations[i][0]
            x_axis_value = axis_values[0]
            y_axis_value = axis_values[1]
            neurons[i].adjust_axis_values(x_axis_value, y_axis_value, self.x_axis_length, self.y_axis_length)

        return neurons

    def create_neurons(self, algorithm, dimensions, number_of_iterations, start_radius, number_of_neurons):
        neurons = []
        x_axis_counter = 0
        y_axis_counter = 0
        for i in range(number_of_neurons):
            neuron = algorithm.create_neuron(dimensions, algorithm,
                                             number_of_iterations, x_axis_counter, y_axis_counter,
                                             start_radius, self.x_axis_length, self.y_axis_length)
            neurons.append(neuron)

            x_axis_counter = x_axis_counter + 1
            if x_axis_counter == self.x_axis_length:
                x_axis_counter = 0
                y_axis_counter = y_axis_counter + 1

        return neurons

    @staticmethod
    def get_neuron_by_grid_coordinate(x_axis, y_axis, neurons):
        for neuron in neurons:
            if neuron.x_axis_counter == x_axis and neuron.y_axis_counter == y_axis:
                return neuron

        print("Es wurde kein Neuron an den grid-stellen: " + str(x_axis) + " und " + str(y_axis) + " gefunden")
        exit()

    def create_new_algorithm(self, algorithm):
        # the algorithm needs to be initialized again, as some carryover effects might occur otherwise
        alg = None
        if type(algorithm) is SOM:
            alg = SOM()
        elif type(algorithm) is NG:
            alg = NG()
        elif type(algorithm) is GNG:
            alg = GNG()
        else:
            print("Unbekannter Algorithmus")
            exit()

        return alg

    def find_neighbourhood(self, current_iteration, best_matching_unit, radius, input_vector):
        # as the neighbourhood is not easily chosen in highdimensional space by the NG,
        # the amount of neurons in the neighbourhood is capped, ensuring that not each iteration only
        # one or all neurons are inside the neighbourhood
        fixed_neighborhood_size = 0
        calculate_amount_of_neurons_in_twodimensional = False
        if isinstance(self.algorithm, NG) and calculate_amount_of_neurons_in_twodimensional:
            fixed_neighborhood_size = NG.calculate_dummy_two_dimensional_neighbouthood_amount(self.number_of_neurons,
                                                                                              current_iteration,
                                                                                              self.number_of_iterations)
        if fixed_neighborhood_size < 1:
            neighbourhood, self.neurons = self.algorithm.find_neighbourhood(self.neurons, best_matching_unit, radius,
                                                                            input_vector)
        else:
            neighbourhood, self.neurons = self.algorithm.find_x_nearest_neighbors(self.neurons, best_matching_unit,
                                                                                  radius, input_vector,
                                                                                  fixed_neighborhood_size)

        if isinstance(self.algorithm, NG) and len(neighbourhood) > int(
                len(self.neurons) * 0.1) and current_iteration * 100 >= self.number_of_iterations:
            smaller_neighbour_list = []
            for neighbour in neighbourhood:
                smaller_neighbour_list.append((neighbour.calculate_distance(best_matching_unit), neighbour))
            smaller_neighbour_list = sorted(smaller_neighbour_list, key=lambda neuron: neuron[0])
            neighbourhood = []
            for i in range(int(len(self.neurons) * 0.1)):
                neighbourhood.append(smaller_neighbour_list[i][1])

        return neighbourhood
