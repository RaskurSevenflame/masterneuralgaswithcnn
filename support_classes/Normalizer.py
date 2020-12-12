import numpy as np


class Normalizer:
    @staticmethod
    def normalize(data):
        if type(data[0][0]) is np.float32:
            new_data = np.zeros([len(data), len(data[0])])
            for column in range(len(data[0])):
                min_value = 0
                max_value = 0

                for row_count in range(len(data)):
                    if data[row_count][column] < min_value:
                        min_value = data[row_count][column]
                    if data[row_count][column] > max_value:
                        max_value = data[row_count][column]

                max_value = max_value + min_value  # min_value can only be negative

                for row_count in range(len(data)):
                    # - min_value scales it to [0, inv]
                    # / max_value scales it to [0, 1]
                    if max_value == 0:
                        new_data[row_count][column] = 0
                    else:
                        new_data[row_count][column] = (data[row_count][column] - min_value) / max_value
            return new_data
        else:
            for column in range(len(data[0][0])):
                min_value = 0
                max_value = 0
                for x_train in data:
                    for row_count in range(len(x_train)):
                        if x_train[row_count][column] < min_value:
                            min_value = x_train[row_count][column]
                        if x_train[row_count][column] > max_value:
                            max_value = x_train[row_count][column]

                    max_value = max_value - min_value  # min_value can only be negative

                for x_train in data:
                    for row_count in range(len(x_train)):
                        # - min_value scales it to [0, inv]
                        # / max_value scales it to [0, 1]
                        if max_value == 0:
                            x_train[row_count][column] = 0
                        else:
                            x_train[row_count][column] = (x_train[row_count][column] - min_value) / max_value

            return data