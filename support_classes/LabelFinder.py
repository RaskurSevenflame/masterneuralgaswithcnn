from errorcalculations.MeasureErrorValue import MeasureErrorValue
from errorcalculations.KruskalShepardError import KruskalShepardError
import math

class LabelFinder:

    """
        finds the fitting label for a neuron in the dataset
    """

    @staticmethod
    def find_fitting_labels_using_grid(neurons, label, data, base, amount_of_different_labels):
        fitting_labels = []

        grid = MeasureErrorValue.generate_label_counting_grid(neurons, data, label, base, amount_of_different_labels)

        array_count = 0
        for neuron in neurons:
            biggest_amount = 0
            for i in range(amount_of_different_labels):
                if grid[neuron.y_axis_counter][neuron.x_axis_counter][i] >= biggest_amount:
                    if len(fitting_labels) <= array_count:
                        fitting_labels.append(i)
                    else:
                        fitting_labels[array_count] = i
                    biggest_amount = grid[neuron.y_axis_counter][neuron.x_axis_counter][i]
            array_count = array_count + 1

        return fitting_labels

    @staticmethod
    def find_fitting_labels(label, data, neurons):
        fitting_labels = []
        dimensions = len(data[0])
        for neuron in neurons:
            fitting_label = None
            closest_data_point_distance = math.inf
            for i in range(len(data)):
                reshaped_vector = data[i].reshape(dimensions, 1)
                distance = KruskalShepardError.calculate_euclidian_distance_between_data_points(reshaped_vector,
                                                                                                neuron.weights)
                if distance < closest_data_point_distance:
                    fitting_label = int(label[i])
                    closest_data_point_distance = distance
            fitting_labels.append(fitting_label)
        return fitting_labels

    @staticmethod
    def old_find_fitting_label(label, data, vector):
        fitting_label = None
        closest_data_point_distance = math.inf
        for i in range(len(data)):
            distance = KruskalShepardError.calculate_euclidian_distance_between_data_points(data[i], vector)
            if distance < closest_data_point_distance:
                fitting_label = int(label[i])
                closest_data_point_distance = distance
        return fitting_label
