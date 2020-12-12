from meta_learning.MetaEvolution import MetaEvolution


class MetaLearningController:

    @staticmethod
    def metalearning(self_organizing_maps, data, label, x_axis_length, y_axis_length, number_of_iterations, random_type, file_to_load, error, amount_of_different_labels):

        meta = MetaEvolution()
        parameter = None
        gamma = 1
        tune_iterations = 25
        number_of_children = 5
        tune_child_iteration = 3
        for organizer in self_organizing_maps:
            algorithm = organizer
            meta.tune_parameter(algorithm, parameter, data, x_axis_length, y_axis_length, number_of_iterations,
                                random_type, gamma, tune_iterations, number_of_children, tune_child_iteration, label,
                                file_to_load, error, amount_of_different_labels)