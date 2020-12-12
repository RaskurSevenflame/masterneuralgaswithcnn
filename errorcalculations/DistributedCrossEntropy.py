import numpy as np
from errorcalculations.MeasureErrorValue import MeasureErrorValue
import math


class DistributedCrossEntropy(MeasureErrorValue):

    """
        Calculates the Cross Entropy for each label seperatly

        furthermore enforces a distribution of main-classes of the neurons
        in comparison to the distribution in the dataset

        additionally if a label is not represented by the neurons, a hughe penalty is added
    """

    def measure_error(self, data, neurons, base, label, amount_of_different_labels, after_dimension_reduction):
        global_error = np.zeros(amount_of_different_labels)
        desired_distribution = np.zeros(amount_of_different_labels)
        achived_distribution = np.zeros(amount_of_different_labels)
        amount_of_labels_not_represented_by_neurons =  0

        grid = self.generate_label_counting_grid(neurons, data, label, base, amount_of_different_labels)

        row_counter = 0
        for row in grid:
            col_counter = 0
            for col in row:
                if np.sum(col) > 0:
                    main_class = 0
                    amount = 0
                    for i in range(amount_of_different_labels):
                        if grid[row_counter][col_counter][i] >= amount:
                            amount = int(grid[row_counter][col_counter][i])
                            main_class = i

                    global_error[main_class] += -1 * np.log(np.max(col) / np.sum(col))
                    achived_distribution[main_class] += 1
                col_counter += 1
            row_counter += 1

        for i in range(amount_of_different_labels):
            # the 5 is corresponding to the calculation in the main class, as the amount of neurons
            # is 1/5 of the datasize
            desired_distribution[i] = list(label).count(i) / 5

        for i in range(amount_of_different_labels):
            desired_numbers = []
            for j in range(int(len(neurons) * 0.02)):
                desired_numbers.append(desired_distribution[i] + j)
                desired_numbers.append(desired_distribution[i] - j)
            if achived_distribution[i] == 0:
                amount_of_labels_not_represented_by_neurons += 1
            elif not (achived_distribution[i] in desired_numbers):
                global_error[i] = abs(achived_distribution[i] - desired_distribution[i]) * global_error[i]

        global_error = np.sum(global_error)
        global_error = global_error * (1 + amount_of_labels_not_represented_by_neurons)

        return global_error

    def get_name(self):
        return "NamePending"
