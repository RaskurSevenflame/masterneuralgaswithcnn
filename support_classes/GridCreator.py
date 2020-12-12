import numpy as np
from support_classes.Direction import Direction
from self_organizing_maps.Base import Base
from self_organizing_maps.Neuron import Neuron
from support_classes.Normalizer import Normalizer


class GridCreator:
    """
      grid creation is not trivial, as the distances are a major factor, between the cells and the neurons themselfs

      in this class different methods are presented:
      closing in: fills the cells of the border first until the hole border is filled, then goes down a layer repeating the process
      inside out: fills the cells in the middle first, resulting in a reverse iteration of "closing in"
      mirroring: starts in each corner and fills the neighbouring cells each iteration. meaning in one iteration up to four cells are filled
      weighting_distances: weights the distance in a none effective way, as a few tests have shown
      comparing_data_point_distances: compares the distance of a neuron in a cell to each other neuron to determine the new neuron for ne neighbouring cell
      classic: goes from left to right, top to down
      classic_plus_point_distances: like classic, while measuring distances between neurons and not cells
      mirroring_and_comparing_data_point_distances: uses the mirroring approach while measuring distances between neurons and not cells
      closest_unit: uses the mirroring approach and only places the closest unit to any currently available neuron
    """

    def __init__(self, neurons):
        self.neurons = neurons
        self.neuron_grid = None

    def create_grid_from_dimension_reduction(self, reduced_data, color, x_axis_length, y_axis_length, mode):
        reduced_data = Normalizer.normalize([reduced_data])[0]
        combined_data = np.concatenate((np.array(reduced_data), np.array(color)), axis=1)

        grid = np.zeros([y_axis_length, x_axis_length, 5])
        for row in grid:
            for col in row:
                col[0] = -1

        self.neuron_grid = [[Neuron for x in range(x_axis_length)] for y in range(y_axis_length)]

        if mode == "closing_in":
            self.closing_in(combined_data, grid, x_axis_length, y_axis_length)
        elif mode == "inside_out":
            self.inside_out(combined_data, grid, x_axis_length, y_axis_length)
        elif mode == "mirroring":
            self.mirroring(combined_data, grid, x_axis_length, y_axis_length)
        elif mode == "weighting_distances":
            self.weighting_distances(combined_data, grid, x_axis_length, y_axis_length)
        elif mode == "comparing_data_point_distances":
            self.comparing_data_point_distances(combined_data, grid, x_axis_length, y_axis_length)
        elif mode == "classic":
            self.classic(combined_data, grid, x_axis_length, y_axis_length)
        elif mode == "classic_plus_point_distances":
            self.classic_plus_point_distances(combined_data, grid, x_axis_length, y_axis_length)
        elif mode == "mirroring_and_comparing_data_point_distances":
            self.mirroring_and_comparing_data_point_distances(combined_data, grid, x_axis_length, y_axis_length)
        elif mode == "closest_unit":
            self.closest_unit(combined_data, grid, x_axis_length, y_axis_length)

        else:
            print("An error occured during selecting modes")
        return grid, self.neurons, self.neuron_grid

    def generate_grid_unit(self, data, grid, row, column, x_value, y_value, x_axis_length, y_axis_length):
        unit, new_data, row_of_the_neuron = self.find_closest_unit(x_value / x_axis_length, y_value / y_axis_length,
                                                                   data)
        for i in range(5):
            grid[column][row][i] = unit[i]

        highdimensional_neuron = self.neurons[row_of_the_neuron]
        highdimensional_neuron.adjust_axis_values(column, row, x_axis_length, y_axis_length)

        self.neuron_grid[column][row] = highdimensional_neuron

        return new_data

    def find_closest_unit(self, x, y, combined_data):
        closest_unit = None
        smallest_distance = 9999  # this will alway be bigger than the first value of squared_distance
        row_counter = 0
        final_row = 0
        for neuron in combined_data:
            squared_distance = (x - neuron[0]) * (x - neuron[0]) + (y - neuron[1]) * (y - neuron[1])
            if smallest_distance > squared_distance:
                closest_unit = [neuron[2], neuron[3], neuron[4], neuron[1], neuron[0]]
                smallest_distance = squared_distance
                final_row = row_counter
            row_counter = row_counter + 1

        new_data = []
        if np.size(combined_data) == 0:
            print("Row: " + str(x) + " Col: " + str(y) + " Data: " + str(combined_data))
        else:
            new_data = np.delete(combined_data, final_row, axis=0)
        return closest_unit, new_data, final_row

    def classic(self, combined_data, grid, x_axis_length, y_axis_length):
        for column in range(0, y_axis_length):
            for row in range(0, x_axis_length):
                combined_data = self.generate_grid_unit(combined_data, grid, row, column, row, column,
                                                        x_axis_length,
                                                        y_axis_length)

    def classic_plus_point_distances(self, combined_data, grid, x_axis_length, y_axis_length):
        last_row = None
        last_column = None

        for column in range(0, y_axis_length):
            for row in range(0, x_axis_length):
                if last_row == None and last_column == None:
                    x_value = 0
                    y_value = 0
                else:
                    x_value = grid[last_column][last_row][4] * x_axis_length
                    y_value = grid[last_column][last_row][3] * y_axis_length

                combined_data = self.generate_grid_unit(combined_data, grid, row, column, x_value, y_value,
                                                        x_axis_length, y_axis_length)

                last_row = row
                last_column = column

    def closing_in(self, combined_data, grid, x_axis_length, y_axis_length):
        diagonal = 0
        changed = True
        while changed:
            changed = False
            row = diagonal
            column = diagonal

            while row + 1 <= x_axis_length - diagonal and np.size(combined_data) > 0:
                changed = True
                combined_data = self.generate_grid_unit(combined_data, grid, row, column, row, column, x_axis_length,
                                                        y_axis_length)
                row = row + 1

            row = diagonal

            while row + 1 <= x_axis_length - diagonal and column + 1 <= (y_axis_length / 2) + 0.5 and np.size(
                    combined_data) > 0:
                changed = True
                combined_data = self.generate_grid_unit(combined_data, grid, row, y_axis_length - column - 1, row,
                                                        y_axis_length - column - 1, x_axis_length, y_axis_length)
                row = row + 1

            row = diagonal
            diagonal = diagonal + 1
            column = diagonal

            while column + 1 <= y_axis_length - diagonal and np.size(combined_data) > 0:
                changed = True
                combined_data = self.generate_grid_unit(combined_data, grid, row, column, row, column, x_axis_length,
                                                        y_axis_length)
                column = column + 1

            column = diagonal

            while column + 1 <= y_axis_length - diagonal and row + 1 <= (x_axis_length / 2) + 0.5 and np.size(
                    combined_data) > 0:
                changed = True
                combined_data = self.generate_grid_unit(combined_data, grid, x_axis_length - row - 1, column,
                                                        x_axis_length - row - 1,
                                                        column, x_axis_length, y_axis_length)
                column = column + 1

    def mirroring(self, combined_data, grid, x_axis_length, y_axis_length):
        diagonal = 0
        changed = True
        while changed:
            changed = False
            row = diagonal
            column = diagonal

            while row + 1 <= (x_axis_length / 2) + 0.5 and np.size(combined_data) > 0:
                changed = True
                combined_data = self.generate_grid_unit(combined_data, grid, row, column, row, column, x_axis_length,
                                                        y_axis_length)
                if x_axis_length - row - 1 >= x_axis_length / 2:
                    combined_data = self.generate_grid_unit(combined_data, grid, x_axis_length - row - 1, column,
                                                            x_axis_length - row, column,
                                                            x_axis_length, y_axis_length)
                row = row + 1

            row = diagonal

            while row + 1 <= (x_axis_length / 2) + 0.5 and column + 1 <= (y_axis_length / 2) + 0.5 and np.size(
                    combined_data) > 0:
                changed = True
                combined_data = self.generate_grid_unit(combined_data, grid, row, y_axis_length - column - 1, row,
                                                        y_axis_length - column - 1, x_axis_length, y_axis_length)
                if x_axis_length - row - 1 >= x_axis_length / 2:
                    combined_data = self.generate_grid_unit(combined_data, grid, x_axis_length - row - 1,
                                                            y_axis_length - column - 1,
                                                            x_axis_length - row, y_axis_length - column - 1,
                                                            x_axis_length,
                                                            y_axis_length)
                row = row + 1

            row = diagonal
            diagonal = diagonal + 1
            column = diagonal

            while column + 1 <= (y_axis_length / 2) + 0.5 and np.size(combined_data) > 0:
                changed = True
                combined_data = self.generate_grid_unit(combined_data, grid, row, column, row, column, x_axis_length,
                                                        y_axis_length)
                if y_axis_length - column - 1 >= y_axis_length / 2:
                    combined_data = self.generate_grid_unit(combined_data, grid, row, y_axis_length - column - 1, row,
                                                            y_axis_length - column,
                                                            x_axis_length, y_axis_length)
                column = column + 1

            column = diagonal

            while column + 1 <= (y_axis_length / 2) + 0.5 and row + 1 <= (x_axis_length / 2) + 0.5 and np.size(
                    combined_data) > 0:
                changed = True
                combined_data = self.generate_grid_unit(combined_data, grid, x_axis_length - row - 1, column,
                                                        row, y_axis_length - column, x_axis_length, y_axis_length)
                if y_axis_length - column - 1 >= y_axis_length / 2:
                    combined_data = self.generate_grid_unit(combined_data, grid, x_axis_length - row - 1,
                                                            y_axis_length - column - 1,
                                                            x_axis_length - row, y_axis_length - column, x_axis_length,
                                                            y_axis_length)

                column = column + 1

    def inside_out(self, combined_data, grid, x_axis_length, y_axis_length):
        untouched_center = [int((y_axis_length / 2) - 0.5), int((x_axis_length / 2) - 0.5)]
        step_size = 1
        direction = Direction.UP
        times_stepped = 2

        combined_data = self.generate_grid_unit(combined_data, grid, untouched_center[1], untouched_center[0],
                                                untouched_center[1], untouched_center[0], x_axis_length,
                                                y_axis_length)

        while (step_size <= x_axis_length or step_size <= y_axis_length) and np.size(combined_data) > 0:
            for times in range(0, times_stepped):
                if np.size(combined_data) == 0:
                    break
                for steps in range(0, step_size):
                    if direction == Direction.DOWN:
                        untouched_center[0] = untouched_center[0] - 1
                    elif direction == Direction.RIGHT:
                        untouched_center[1] = untouched_center[1] + 1
                    elif direction == Direction.UP:
                        untouched_center[0] = untouched_center[0] + 1
                    elif direction == Direction.LEFT:
                        untouched_center[1] = untouched_center[1] - 1
                    else:
                        print("An Error occured while stepping in the direction of the enum")
                        exit()

                    if not (untouched_center[0] < 0 or untouched_center[0] >= y_axis_length or untouched_center[
                        1] < 0 or untouched_center[1] >= x_axis_length):
                        combined_data = self.generate_grid_unit(combined_data, grid, untouched_center[1],
                                                                untouched_center[0],
                                                                untouched_center[1], untouched_center[0],
                                                                x_axis_length,
                                                                y_axis_length)

                if direction == Direction.UP:
                    direction = Direction.RIGHT
                elif direction == Direction.RIGHT:
                    direction = Direction.DOWN
                elif direction == Direction.DOWN:
                    direction = Direction.LEFT
                elif direction == Direction.LEFT:
                    direction = Direction.UP
                else:
                    print("An Error occured while switching the enum direction")

            step_size = step_size + 1

    def weighting_distances(self, combined_data, grid, x_axis_length, y_axis_length):
        max_stepsize = max(x_axis_length, y_axis_length)

        beta = 0.7

        for stepsize in range(1, max_stepsize + 1):
            for step in range(0, stepsize):
                row = step
                column = stepsize - 1 - step
                if row + 1 <= (x_axis_length / 2) + 0.5 and column + 1 <= (y_axis_length / 2) + 0.5:
                    if row + 1 <= (x_axis_length / 2) + 0.5:
                        combined_data = self.generate_grid_unit(combined_data, grid, row, column,
                                                                row * beta,
                                                                column * beta, x_axis_length,
                                                                y_axis_length)
                    if x_axis_length - row - 1 >= x_axis_length / 2:
                        combined_data = self.generate_grid_unit(combined_data, grid, x_axis_length - row - 1, column,
                                                                (x_axis_length - row) * beta,
                                                                column * beta, x_axis_length,
                                                                y_axis_length)
                    if row + 1 <= (x_axis_length / 2) + 0.5 and y_axis_length - column - 1 >= y_axis_length / 2:
                        combined_data = self.generate_grid_unit(combined_data, grid, row, y_axis_length - column - 1,
                                                                row * beta,
                                                                (y_axis_length - column) * beta,
                                                                x_axis_length, y_axis_length)
                    if x_axis_length - row - 1 >= x_axis_length / 2 and y_axis_length - column - 1 >= y_axis_length / 2:
                        combined_data = self.generate_grid_unit(combined_data, grid, x_axis_length - row - 1,
                                                                y_axis_length - column - 1,
                                                                (x_axis_length - row) * beta,
                                                                (y_axis_length - column) * beta,
                                                                x_axis_length, y_axis_length)

    def comparing_data_point_distances(self, combined_data, grid, x_axis_length, y_axis_length):
        row = 0
        column = 0

        combined_data = self.generate_grid_unit(combined_data, grid, row, column,
                                                row / x_axis_length,
                                                column / y_axis_length, x_axis_length,
                                                y_axis_length)

        max_stepsize = max(x_axis_length, y_axis_length) * 2

        for stepsize in range(1, max_stepsize + 1):
            for step in range(0, stepsize):
                for neighbours in range(0, 2):
                    if neighbours == 0:
                        row = step + 1
                        column = stepsize - 1 - step
                    else:
                        row = step
                        column = stepsize - step

                    if row + 1 <= x_axis_length and column + 1 <= y_axis_length:
                        if grid[column][row] == 10:
                            combined_data = self.generate_grid_unit(combined_data, grid, row, column,
                                                                    grid[stepsize - 1 - step][step][4] * x_axis_length,
                                                                    grid[stepsize - 1 - step][step][3] * y_axis_length,
                                                                    x_axis_length,
                                                                    y_axis_length)

    def mirroring_and_comparing_data_point_distances(self, combined_data, grid, x_axis_length, y_axis_length):
        # getting the corners fixed
        combined_data = self.generate_grid_unit(combined_data, grid, 0, 0, 0, 0, x_axis_length, y_axis_length)
        combined_data = self.generate_grid_unit(combined_data, grid, x_axis_length - 1, 0, x_axis_length, 0,
                                                x_axis_length,
                                                y_axis_length)
        combined_data = self.generate_grid_unit(combined_data, grid, 0, y_axis_length - 1, 0, y_axis_length,
                                                x_axis_length,
                                                y_axis_length)
        combined_data = self.generate_grid_unit(combined_data, grid, x_axis_length - 1, y_axis_length - 1,
                                                x_axis_length,
                                                y_axis_length, x_axis_length, y_axis_length)

        max_stepsize = max(x_axis_length, y_axis_length)

        for stepsize in range(1, max_stepsize + 1):
            for step in range(0, stepsize):
                for neighbours in range(0, 2):
                    if neighbours == 0:
                        row = step + 1
                        column = stepsize - 1 - step
                    else:
                        row = step
                        column = stepsize - step

                    last_row = step
                    last_column = stepsize - 1 - step

                    if row + 1 <= (x_axis_length / 2) + 0.5 and column + 1 <= (y_axis_length / 2) + 0.5:
                        combined_data = self.check_and_create(column, row, grid, combined_data, last_row, last_column,
                                                              x_axis_length, y_axis_length)

                for neighbours in range(0, 2):
                    if neighbours == 0:
                        row = x_axis_length - 1 - (step + 1)
                        column = stepsize - 1 - step
                    else:
                        row = x_axis_length - 1 - step
                        column = stepsize - step

                    last_row = x_axis_length - 1 - step
                    last_column = stepsize - 1 - step

                    if row >= x_axis_length / 2 and column + 1 <= (y_axis_length / 2) + 0.5:
                        combined_data = self.check_and_create(column, row, grid, combined_data, last_row, last_column,
                                                              x_axis_length, y_axis_length)

                for neighbours in range(0, 2):
                    if neighbours == 0:
                        row = step + 1
                        column = y_axis_length - 1 - (stepsize - 1 - step)
                    else:
                        row = step
                        column = y_axis_length - 1 - (stepsize - step)

                    last_row = step
                    last_column = y_axis_length - 1 - (stepsize - 1 - step)

                    if row + 1 <= (x_axis_length / 2) + 0.5 and column >= y_axis_length / 2:
                        combined_data = self.check_and_create(column, row, grid, combined_data, last_row, last_column,
                                                              x_axis_length, y_axis_length)

                for neighbours in range(0, 2):
                    if neighbours == 0:
                        row = x_axis_length - 1 - (step + 1)
                        column = y_axis_length - 1 - (stepsize - 1 - step)
                    else:
                        row = x_axis_length - 1 - step
                        column = y_axis_length - 1 - (stepsize - step)

                    last_row = x_axis_length - 1 - step
                    last_column = y_axis_length - 1 - (stepsize - 1 - step)

                    if row >= x_axis_length / 2 and column >= y_axis_length / 2:
                        combined_data = self.check_and_create(column, row, grid, combined_data, last_row, last_column,
                                                              x_axis_length, y_axis_length)

    def closest_unit(self, combined_data, grid, x_axis_length, y_axis_length):

        """
        this method is not finished, as all other methods did not achive better results
        and a discussion with Kramer and Elend resulted in ending the search for better grid creations
        this code is left behind here to showcase the most promising approach so far

        the approach was to first fix the corners and afterwards only insert the most promising neighbouring-neuron,
        regardless of it's position
        """
        # getting the corners fixed
        combined_data = self.generate_grid_unit(combined_data, grid, 0, 0, 0, 0, x_axis_length, y_axis_length)
        combined_data = self.generate_grid_unit(combined_data, grid, x_axis_length - 1, 0, x_axis_length, 0,
                                                x_axis_length,
                                                y_axis_length)
        combined_data = self.generate_grid_unit(combined_data, grid, 0, y_axis_length - 1, 0, y_axis_length,
                                                x_axis_length,
                                                y_axis_length)
        combined_data = self.generate_grid_unit(combined_data, grid, x_axis_length - 1, y_axis_length - 1,
                                                x_axis_length,
                                                y_axis_length, x_axis_length, y_axis_length)

        while len(combined_data) > 0:
            pass

    def check_and_create(self, column, row, grid, combined_data, last_row, last_column, x_axis_length, y_axis_length):
        if grid[column][row][0] == -1:
            combined_data = self.generate_grid_unit(combined_data, grid, row, column,
                                                    grid[last_column][last_row][4] * x_axis_length,
                                                    grid[last_column][last_row][3] * y_axis_length,
                                                    x_axis_length, y_axis_length)
        return combined_data
