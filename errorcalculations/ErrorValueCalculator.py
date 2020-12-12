from errorcalculations.KruskalShepardError import KruskalShepardError
from errorcalculations.CrossEntropy import CrossEntropy
from errorcalculations.ClassScatterIndex import ClassScatterIndex
from errorcalculations.GlobalEuclideanError import GlobalEuclideanError
from errorcalculations.MinorClassOccurrence import MinorClassOccurrence
from errorcalculations.DistributedCrossEntropy import DistributedCrossEntropy
from self_organizing_maps.Base import Base
from support_classes.Loader import Loader
import numpy as np


class ErrorValueCalculator:
    """
        Measures each error-value of the neurons
    """

    def calculate_error_values(self, files, dataset_size, self_organizing_maps, x_axis_length, y_axis_length,
                               number_of_iterations, start_learning_rate,
                               start_radius_multiplikator, end_learning_rate, random_type, amount_of_different_labels,
                               after_dimension_reduction, optimized):
        for file_to_load in files:
            loader = Loader()
            data, label = loader.load_dataset(dataset_size, file_to_load)

            if not optimized:
                self.calculate_different_error_measures(optimized, data, label, self_organizing_maps, loader,
                                                        dataset_size, file_to_load, x_axis_length, y_axis_length,
                                                        number_of_iterations, start_learning_rate,
                                                        start_radius_multiplikator, end_learning_rate, random_type,
                                                        amount_of_different_labels, after_dimension_reduction)
            else:
                self.calculate_different_error_measures(optimized, data, label, self_organizing_maps, loader,
                                                        dataset_size, file_to_load, x_axis_length, y_axis_length,
                                                        number_of_iterations, start_learning_rate,
                                                        start_radius_multiplikator, end_learning_rate, random_type,
                                                        amount_of_different_labels, after_dimension_reduction)

    @staticmethod
    def calculate_different_error_measures(optimized, data, label, self_organizing_maps, loader, dataset_size,
                                           file_to_load, x_axis_length, y_axis_length, number_of_iterations,
                                           start_learning_rate, start_radius_multiplikator, end_learning_rate,
                                           random_type, amount_of_different_labels, after_dimension_reduction):
        for algorithm in self_organizing_maps:
            neuron_information = np.asarray(
                loader.load_neuron_informations(dataset_size, file_to_load, algorithm, optimized))

            base = Base(x_axis_length, y_axis_length, algorithm, data, number_of_iterations, start_learning_rate,
                        start_radius_multiplikator, end_learning_rate, random_type)

            neurons = base.generate_neurons_from_neuron_information(neuron_information, algorithm,
                                                                    len(neuron_information[0][1]), number_of_iterations,
                                                                    start_radius_multiplikator)

            error_measures = [GlobalEuclideanError(), MinorClassOccurrence(), ClassScatterIndex(),
                              KruskalShepardError(), CrossEntropy(), DistributedCrossEntropy()]
            for error in error_measures:
                error_value = error.measure_error(data, neurons, base, label, amount_of_different_labels,
                                                  after_dimension_reduction)

                if optimized:
                    print(
                        algorithm.get_name() + "with" + file_to_load + " dataset and optimized parameter, has an error value of: " + str(
                            error_value) + " of the error_type: " + error.get_name())
                else:
                    print(
                        algorithm.get_name() + "with" + file_to_load + " dataset has an error value of: " + str(
                            error_value) + " of the error_type: " + error.get_name())
