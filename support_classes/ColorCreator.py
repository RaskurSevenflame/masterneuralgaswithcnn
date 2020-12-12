from support_classes.LabelFinder import LabelFinder

class ColorCreator:
    @staticmethod

    def get_color_by_label(single_label):
        # encodes into RGB values
        if single_label == 0:
            return [1, 0, 0]
        elif single_label == 1:
            return [0, 1, 0.5]
        elif single_label == 2:
            return [0.5, 1, 0]
        elif single_label == 3:
            return [1, 0, 0.5]
        elif single_label == 4:
            return [1, 0.5, 0]
        elif single_label == 5:
            return [0, 0, 1]
        elif single_label == 6:
            return [0, 1, 1]
        elif single_label == 7:
            return [0, 1, 0]
        elif single_label == 8:
            return [1, 1, 0]
        elif single_label == 9:
            return [1, 0, 1]
        else:
            return [1, 1, 1]

    @staticmethod
    def get_color(neurons, label, data, base, amount_of_different_labels):
        # returns the color corresponding to the label

        # calculates the label on the grid, corresponding to all vectors that are close to it
        # if false the label corresponds to the closest vector in the dataset
        use_grid_to_find_label = False
        colors = []
        if len(neurons[0].weights) == 3:
            for neuron in neurons:
                colors.append(neuron.weights)
        else:
            if use_grid_to_find_label:
                fitting_labels = LabelFinder.find_fitting_labels_using_grid(neurons, label, data, base, amount_of_different_labels)
            else:
                fitting_labels = LabelFinder.find_fitting_labels(label, data, neurons)
            for fitting_label in fitting_labels:
                color = ColorCreator.get_color_by_label(fitting_label)
                colors.append(color)
        return colors

    @staticmethod
    def old_get_color(weights, label, data):
        colors = []
        if len(weights) == 3:
            colors = weights
        else:
            for color in weights:
                single_label = LabelFinder.old_find_fitting_label(label, data, color)
                color = ColorCreator.get_color_by_label(single_label)
                colors.append(color)
        return colors
