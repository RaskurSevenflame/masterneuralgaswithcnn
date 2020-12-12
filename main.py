from self_organizing_maps.self_organizing_map.SOM import SOM
from self_organizing_maps.neural_gas.NG import NG
from self_organizing_maps.growing_neural_gas.GNG import GNG
from meta_learning.MetaLearningController import MetaLearningController
import math
from errorcalculations.ErrorValueCalculator import ErrorValueCalculator
from dimension_reduction.DimensionReductionController import DimensionReductionController
from self_organizing_maps.TrainSelfOrganizingMap import TrainSelfOrganizingMap
from support_classes.Loader import Loader
from dimension_reduction.DimensionReduction import DimensionReduction
from errorcalculations.CrossEntropy import CrossEntropy
from errorcalculations.MinorClassOccurrence import MinorClassOccurrence
import argparse

parser = argparse.ArgumentParser()

"""
    These parameter control which parts of the modular elements are used in a run
    Convolution and Meta Learning are used befor the Self-Organizing Maps
    optimized referrers to parameter and values tuned by the mtea learning
    specific and train_all either train all Self-Organizing Maps or the selected one below in the variable "algorithm"
    weights_after_reduction calculates CSI und KSE for NG and GNG as it's not possible befor
    error_calc measures the errors of the trained neurons
    reduction and opti_reduction use t-SNE to create a grid in 2D space; opti_reduction uses the meta learning values of the neuron weights
    only_tsne uses the dataset directly on t_sne and stores the plot
    
    it is recommendet so run generate_convolution_data, train_all_with_untrained_parameter, dimension_reduction 
    and calculate_errors in one go, to get all needed results
"""
parser.add_argument("-conv", "--generate_convolution_data", action="store_true")
parser.add_argument("-meta", "--meta_training", action="store_true")
parser.add_argument("-specific", "--train_a_specific_algorithm", action="store_true")
parser.add_argument("-optimized", "--use_optimized_parameter", action="store_true")
parser.add_argument("-train_all", "--train_all_with_untrained_parameter", action="store_true")
parser.add_argument("-reduction", "--dimension_reduction", action="store_true")
parser.add_argument("-opti_reduction", "--use_optimized_parameter_for_dimensionreduction", action="store_true")
parser.add_argument("-weights_after_reduction", "--use_neuron_weights_after_dimensionredution_for_error_calculation",
                    action="store_true")
parser.add_argument("-error_calc", "--calculate_errors", action="store_true")
parser.add_argument("-only_tsne", "--showcase_tsne", action="store_true")

args = parser.parse_args()

self_organizing_maps = [SOM(), NG(), GNG()]

files = ["mnistwith0Layer", "mnistwith1Layer", "mnistwith2Layer", "mnistwith3Layer", "cifar10with0Layer",
         "cifar10with1Layer", "cifar10with2Layer", "cifar10with3Layer"]

generate_convolution_data = args.generate_convolution_data
meta_training = args.meta_training
train_a_specific_algorithm = args.train_a_specific_algorithm
use_optimized_parameter = args.use_optimized_parameter
train_all_with_untrained_parameter = args.train_all_with_untrained_parameter
dimension_reduction = args.dimension_reduction
use_optimized_parameter_for_dimensionreduction = args.use_optimized_parameter_for_dimensionreduction
use_neuron_weights_after_dimensionredution_for_error_calculation = args.use_neuron_weights_after_dimensionredution_for_error_calculation
calculate_errors = args.calculate_errors
showcase_tsne = args.showcase_tsne

if generate_convolution_data:
    from convolution.ConvolutionController import ConvolutionController


def main():
    amount_of_different_labels = 10

    dataset_size = 500  # 1000, 1500
    specific_algorithm = self_organizing_maps[0]
    # generating the number of iterations adaptively has proven to generate better results than a fixed number
    adaptive_number_of_iterations_and_neurons = True

    scale_iterations_per_datapoint = 100
    scale_number_of_neurons_per_datapoint = 1 / 5
    # if not adaptive this settings will be used
    number_of_iterations = 1000

    # since some of the newer creations do not go well with odd numbers and strongly different numbers it is
    # recommended to not trust the results after changing these values rapidly
    x_axis_length = 10
    y_axis_length = 10

    # this drops linearly over the duration of the iteration, if not changed in the corresponding algorithm
    start_learning_rate = 0.5
    end_learning_rate = 0.05

    # the grid has always the length of 1, this calculates the assumed radius of a neuron in the given dimensionality
    start_radius_multiplikator = 0.5

    random_types = ["not_random", "cicle", ""]
    random_type = random_types[0]

    error_values_for_meta_learning = [CrossEntropy(), MinorClassOccurrence()]

    if generate_convolution_data:
        datasets = ["mnist", "cifar10"]  # "None"
        for set in datasets:
            cnn_controller = ConvolutionController()
            cnn_controller.generate_cnn_data(set, dataset_size)

    if meta_training or use_optimized_parameter or train_a_specific_algorithm or train_all_with_untrained_parameter:
        loader = Loader()
        trainer = TrainSelfOrganizingMap()
        for file_to_load in files:
            data, label = loader.load_dataset(dataset_size, file_to_load)

            if adaptive_number_of_iterations_and_neurons:
                number_of_iterations = scale_iterations_per_datapoint * dataset_size
                number_of_neurons = dataset_size * scale_number_of_neurons_per_datapoint
                x_axis_length = int(math.sqrt(number_of_neurons))
                y_axis_length = int(number_of_neurons / x_axis_length)

            if meta_training:
                meta_learning_controller = MetaLearningController()
                meta_learning_controller.metalearning(self_organizing_maps, data, label, x_axis_length, y_axis_length,
                                                      number_of_iterations, random_type, file_to_load, error_values_for_meta_learning,
                                                      amount_of_different_labels)

            if use_optimized_parameter:
                for organizer in self_organizing_maps:
                    optimized = True
                    parameter = loader.load_parameter(file_to_load, organizer, data)
                    trainer.train_self_organizing_maps_algorithm(data, label, organizer, x_axis_length, y_axis_length,
                                                                 number_of_iterations, float(parameter[0]),
                                                                 float(parameter[1]), 1 / float(parameter[2]),
                                                                 random_type, optimized, file_to_load,
                                                                 amount_of_different_labels)

            if train_a_specific_algorithm:
                trainer.train_self_organizing_maps_algorithm(data, label, specific_algorithm, x_axis_length, y_axis_length,
                                                             number_of_iterations, start_learning_rate,
                                                             start_radius_multiplikator, end_learning_rate, random_type,
                                                             False, file_to_load, amount_of_different_labels)

            if train_all_with_untrained_parameter:
                for organizer in self_organizing_maps:
                    trainer.train_self_organizing_maps_algorithm(data, label, organizer, x_axis_length, y_axis_length,
                                                                 number_of_iterations, start_learning_rate,
                                                                 start_radius_multiplikator,
                                                                 end_learning_rate, random_type, False, file_to_load,
                                                                 amount_of_different_labels)

    if dimension_reduction:
        reduction = DimensionReductionController()
        reduction.reduce_data(self_organizing_maps, x_axis_length, y_axis_length, files, dataset_size,
                              use_optimized_parameter_for_dimensionreduction, number_of_iterations, start_learning_rate,
                              start_radius_multiplikator, end_learning_rate, random_type, amount_of_different_labels)

    if calculate_errors:
        error_calculator = ErrorValueCalculator()
        error_calculator.calculate_error_values(files, dataset_size, self_organizing_maps, x_axis_length, y_axis_length,
                                                number_of_iterations, start_learning_rate,
                                                start_radius_multiplikator, end_learning_rate, random_type,
                                                amount_of_different_labels,
                                                use_neuron_weights_after_dimensionredution_for_error_calculation,
                                                use_optimized_parameter)

    if showcase_tsne:
        t_sne = DimensionReduction()
        loader = Loader()
        file_to_load = files[7]
        data, label = loader.load_dataset(dataset_size, file_to_load)
        reshaped_label = label.reshape(len(data))
        t_sne.t_sne(data, reshaped_label)


main()
