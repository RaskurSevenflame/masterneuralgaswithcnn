import numpy as np
from errorcalculations.MeasureErrorValue import MeasureErrorValue
from self_organizing_maps.self_organizing_map.SOM import SOM


class KruskalShepardError(MeasureErrorValue):
    """
        measures the distances that are present in the highdimensional space
        and compares them to the low dimensional space
    """

    def measure_error(self, data, neurons, base, label, amount_of_different_labels, after_dimension_reduction):
        dimensions = len(data[0])
        if after_dimension_reduction or type(base.algorithm) is SOM:
            high_dimensional_distance_array = []
            low_dimensional_distance_array = []

            for vector1 in data:
                input_vector1 = np.asarray(vector1)
                reshaped_vector1 = input_vector1.reshape(dimensions, 1)
                bmu1 = base.find_best_matching_unit(reshaped_vector1, neurons)

                high_dimensional_array = np.zeros(len(data))
                low_dimensional_array = np.zeros(len(data))
                array_count = 0

                for vector2 in data:
                    input_vector2 = np.asarray(vector2)
                    reshaped_vector2 = input_vector2.reshape(dimensions, 1)
                    bmu2 = base.find_best_matching_unit(reshaped_vector2, neurons)

                    high_dimensional_array[array_count] = self.calculate_euclidian_distance_between_data_points(
                        reshaped_vector1, reshaped_vector2)
                    low_dimensional_array[array_count] = self.calculate_distance_on_grid(bmu1, bmu2)

                    array_count += 1

                high_dimensional_distance_array.append(np.asarray(high_dimensional_array))
                low_dimensional_distance_array.append(np.asarray(low_dimensional_array))

            high_dimensional_distance_array = high_dimensional_distance_array / np.max(high_dimensional_distance_array)
            low_dimensional_distance_array = low_dimensional_distance_array / np.max(low_dimensional_distance_array)

            distance_array = high_dimensional_distance_array - low_dimensional_distance_array

            distance = np.linalg.norm(distance_array, 'fro')
            distance = distance ** 2
            # distance = np.sqrt(np.sum(distance_array))
            return distance / ((len(data) * len(data)) - len(data))

        else:
            return "no comparable dimension"

    @staticmethod
    def calculate_euclidian_distance_between_data_points(vector1, vector2):
        calc = (vector1 - vector2) ** 2
        value = np.sum(calc, axis=0)
        if len(value) > 1:
            print("An Error occured during the calculation of the distance")
        # return float(value)
        # print(float(np.sqrt(value)))
        return float(np.sqrt(value))

    @staticmethod
    def calculate_distance_on_grid(neuron1, neuron2):
        return np.sqrt(((neuron1.x_center - neuron2.x_center) * (neuron1.x_center - neuron2.x_center)) + (
                (neuron1.y_center - neuron2.y_center) * (neuron1.y_center - neuron2.y_center)))

    def get_name(self):
        return "KruskalShepardError"
