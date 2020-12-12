from errorcalculations.MeasureErrorValue import MeasureErrorValue
import math
from self_organizing_maps.neural_gas.NG import NG
from self_organizing_maps.growing_neural_gas.GNG import GNG


class ClassScatterIndex(MeasureErrorValue):

    """
        Calculates the amount of clusters in the grid
    """

    def measure_error(self, data, neurons, base, label, amount_of_different_labels, after_dimension_reduction):

        # checks if NG or GNG are unreduced, because they can't calculate the CSI if they are
        if not after_dimension_reduction and (isinstance(base.algorithm, NG) or isinstance(base.algorithm, GNG)):
            return "guessing"

        global_error = 0
        coordinate_list = []
        list_of_remaining_grid_coordinates = []
        grid = self.generate_label_counting_grid(neurons, data, label, base, amount_of_different_labels)
        if grid is None:
            return math.inf

        # checks for each label, how many clusters are found inside the grid
        for label_value in range(0, amount_of_different_labels):
            for i in range(base.y_axis_length):
                for j in range(base.x_axis_length):
                    list_of_remaining_grid_coordinates.append([i, j])

            while len(list_of_remaining_grid_coordinates) > 0:
                new_label_coordinate = list_of_remaining_grid_coordinates[0]
                if grid[new_label_coordinate[0]][new_label_coordinate[1]][label_value] < 1:
                    list_of_remaining_grid_coordinates.remove(new_label_coordinate)
                else:
                    coordinate_list.append(new_label_coordinate)
                    global_error += 1

                    for coordinate in coordinate_list:
                        coordinate_list.remove(coordinate)
                        if coordinate in list_of_remaining_grid_coordinates:
                            list_of_remaining_grid_coordinates.remove(coordinate)

                            neighbouring_x_y_coordinates = [[coordinate[0] + 1, coordinate[1]],
                                                            [coordinate[0], coordinate[1] - 1],
                                                            [coordinate[0] - 1, coordinate[1]],
                                                            [coordinate[0], coordinate[1] + 1]]

                            for coordinate in neighbouring_x_y_coordinates:
                                new_x = coordinate[1]
                                new_y = coordinate[0]
                                if 0 <= new_x < len(grid[0]) and 0 <= new_y < len(grid):
                                    if [new_y, new_x] in list_of_remaining_grid_coordinates:
                                        if grid[new_y][new_x][label_value] > 0:
                                            coordinate_list.append([new_y, new_x])

        return global_error / amount_of_different_labels

    def get_name(self):
        return "ClassScatterIndex"
