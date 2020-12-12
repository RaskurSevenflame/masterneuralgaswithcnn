import numpy as np
from dimension_reduction.DimensionReduction import DimensionReduction
from support_classes.GridCreator import GridCreator
from support_classes.CreatePlot import CreatePlot
from support_classes.Loader import Loader
from self_organizing_maps.Base import Base
from errorcalculations.KruskalShepardError import KruskalShepardError
from self_organizing_maps.self_organizing_map.SOM import SOM
from errorcalculations.ClassScatterIndex import ClassScatterIndex
from support_classes.ColorCreator import ColorCreator
from support_classes.LabelFinder import LabelFinder


class DimensionReductionController:

    def reduce_data(self, algorithms, x_axis_length, y_axis_length, files, dataset_size, optimized,
                    number_of_iterations, start_learning_rate,
                    start_radius_multiplicator, end_learning_rate, random_type, amount_of_different_labels):
        """
        :param algorithms: used algorithms of the Self-Organizing Maps type
        :param x_axis_length:
        :param y_axis_length:
        :param files: the datasets that are used
        :param dataset_size: size of the current dataset
        :param optimized: true if the parameter were produced by meta learning
        :param number_of_iterations:
        :param start_learning_rate:
        :param start_radius_multiplicator:
        :param end_learning_rate:
        :param random_type: states in which way new vectors are chosen
        :param amount_of_different_labels:
        :return:
        """

        loader = Loader()

        for file in files:
            for algorithm in algorithms:
                data, label = loader.load_dataset(dataset_size, file)

                base = Base(x_axis_length, y_axis_length, algorithm, data, number_of_iterations, start_learning_rate,
                            start_radius_multiplicator, end_learning_rate, random_type)

                neuron_information = np.asarray(
                    loader.load_neuron_informations(dataset_size, file, algorithm, optimized))
                neurons = base.generate_neurons_from_neuron_information(neuron_information, algorithm,
                                                                        len(neuron_information[0][1]),
                                                                        number_of_iterations,
                                                                        start_radius_multiplicator)

                # plots the SOM without dimension reduction
                if isinstance(algorithm, SOM):
                    colors = ColorCreator.get_color(neurons, label, data, base, amount_of_different_labels)
                    grid = np.zeros([y_axis_length, x_axis_length, 3])
                    for i in range(len(neurons)):
                        for j in range(3):
                            grid[neurons[i].y_axis_counter][neurons[i].x_axis_counter][j] = colors[i][j]
                    CreatePlot.create_plot_from_reduced_data(x_axis_length, y_axis_length, grid, "SOM - without reduction")

                neuron_weights = []
                for neuron in neurons:
                    neuron_weights.append(neuron.weights)

                # performs the dimension-reduction on t-SNE
                t_sne_result, color = self.use_dimensionreduction(label, data, neurons, base, amount_of_different_labels)

                """
                    Please note, that these modes will generate different grids.
                    As these do not produce acceptable results they are not used at the moment
                """
                modes = ["classic", "closing_in", "mirroring", "inside_out", "weighting_distances",
                         "comparing_data_point_distances", "classic_plus_point_distances",
                         "mirroring_and_comparing_data_point_distances"]
                modes = ["classic_plus_point_distances"]

                # creates the grids of the chosen modes after the dimensionreduction
                for mode in modes:
                    grid_creator = GridCreator(neurons)
                    grid, sorted_neurons, high_dimensional_neuron_grid = grid_creator.create_grid_from_dimension_reduction(
                        t_sne_result, color, x_axis_length,
                        y_axis_length,
                        mode)

                    kruskal = KruskalShepardError()
                    error = kruskal.measure_error(data, sorted_neurons, base, label, amount_of_different_labels, True)
                    print(algorithm.get_name() + " with kruskal: " + str(error) + " on " + file)
                    class_scatter = ClassScatterIndex()
                    error = class_scatter.measure_error(data, sorted_neurons, base, label, amount_of_different_labels, True)
                    print(algorithm.get_name() + " with class_scatter: " + str(error) + " on " + file)

                    CreatePlot.create_plot_from_reduced_data(x_axis_length, y_axis_length, grid, mode)

    def use_dimensionreduction(self, label, data, neurons, base, amount_of_different_labels):
        color = ColorCreator.get_color(neurons, label, data, base, amount_of_different_labels)

        reduction = DimensionReduction()
        neuron_label = LabelFinder.find_fitting_labels(label, data, neurons)
        t_sne_result = reduction.reduce_with_neuron_weights(neurons, neuron_label, data, base,
                                                            amount_of_different_labels)

        return t_sne_result, color
