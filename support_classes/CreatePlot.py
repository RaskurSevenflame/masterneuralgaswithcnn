import matplotlib.pyplot as plt
import numpy as np


class CreatePlot:

    """
        Creates the Plot fromt the grid
    """
    @staticmethod
    def create_plot_from_reduced_data(x_axis_length, y_axis_length, grid, mode):
        fig = plt.figure(figsize=(15, 8))
        fig.subtitle("Mode: %s" % (mode), fontsize=14)

        red = []
        green = []
        blue = []
        for column in grid:
            for row in column:
                red.append(row[0])
                green.append(row[1])
                blue.append(row[2])

        red = np.asarray(red).reshape(x_axis_length, y_axis_length)
        green = np.asarray(green).reshape(x_axis_length, y_axis_length)
        blue = np.asarray(blue).reshape(x_axis_length, y_axis_length)
        c = np.dstack([red, green, blue])

        plt.imshow(c, interpolation='nearest')
        plt.show()
