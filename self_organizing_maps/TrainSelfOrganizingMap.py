import pickle
from self_organizing_maps.Base import Base
from errorcalculations.DistributedCrossEntropy import DistributedCrossEntropy
from errorcalculations.CrossEntropy import CrossEntropy
import numpy as np
from self_organizing_maps.growing_neural_gas.GNG import GNG


class TrainSelfOrganizingMap:

    @staticmethod
    def train_self_organizing_maps_algorithm(data, label, algorithm, x_axis_length, y_axis_length, number_of_iterations,
                                             start_learning_rate, start_radius_multiplikator, end_learning_rate,
                                            random_type, optimized, file_to_load, amount_of_differen_labels):
        # trains the algorithm and saves its neuron-information in a pickle file in /saves
        base = Base(x_axis_length, y_axis_length, algorithm, data, number_of_iterations, start_learning_rate,
                    start_radius_multiplikator, end_learning_rate, random_type)
        neurons = base.train()

        error = DistributedCrossEntropy()
        error_value = error.measure_error(data, neurons, base, label, amount_of_differen_labels, False)

        print(algorithm.get_name() + " has an error value of: " + str(error_value) + " " + error.get_name())

        information = []
        for neuron in neurons:
            information.append([[neuron.x_axis_counter, neuron.y_axis_counter], neuron.weights])

        name_tag = "NeuronWeights" + "_Datasetsize" + str(len(data))
        if optimized:
            name_tag = name_tag + "optimized"

        file_name = file_to_load + algorithm.get_name() + name_tag
        pickle.dump(information, open(
            "saves/" + file_name,
            "wb"))
