from self_organizing_maps.Base import Base
import numpy as np
import pickle
import math


class MetaEvolution(object):

    def tune_parameter(self, algorithm, parameter, data, x_axis_length, y_axis_length, number_of_iterations,
                       random_type, gamma, tune_iterations, number_of_children, tune_child_iteration, label,
                       file_to_load, error, amount_of_different_labels):

        if parameter is None:
            parameter = [0.5, 0.5, 20]

        # these are used to train on different error-values at once and try to improve all of them
        error_values = np.ones(len(error)) * math.inf
        new_error_values = np.ones(len(error)) * math.inf
        error_increase = np.ones(len(error)) * math.inf

        rho = math.sqrt(len(data) + 1)

        # the following line can be used to train the different random_types,
        # which is not done here as most would place randomness into the calculation,
        # which is not recommended in meta learning,
        # as the variance may result in bad parameters with a single good error-value
        # for random_type in random_types:
        results = []
        for tune_iteration in range(tune_iterations):
            if len(results) > 0:
                results = [results[0]]
            for child in range(number_of_children):
                print("processing the child-nr: " + str(child) + " of tune-iteration: " + str(tune_iteration))
                smallest_error_value = math.inf
                best_parameter = parameter
                new_gamma = gamma
                for iteration in range(tune_child_iteration):
                    new_parameter = self.gaussian_mutation(best_parameter, new_gamma)

                    base = Base(x_axis_length, y_axis_length, algorithm, data, number_of_iterations,
                                new_parameter[0],
                                new_parameter[1], (1 / new_parameter[2]), random_type)

                    neurons = base.train()

                    for i in range(len(error)):
                        new_error_values[i] = error[i].measure_error(data, neurons, base, label,
                                                                     amount_of_different_labels, False)
                        error_increase[i] = new_error_values[i] / error_values[i]

                    error_difference = 0
                    for error_change in error_increase:
                        if error_change <= 1:
                            error_difference += 1 - error_change
                        else:
                            error_difference -= 1.5 * (error_change - 1)

                    if error_difference >= 0:
                        best_parameter = new_parameter
                        for i in range(len(error_values)):
                            error_values[i] = new_error_values[i]
                        smallest_error_value = np.sum(error_values)

                    print(str(error_values) + " " + str(new_error_values))

                    new_gamma = new_gamma * math.exp((error_difference >= 0) - 1 / 5) ** (1 / rho)

                result = np.append([smallest_error_value, new_gamma], best_parameter)
                results.append(result)

            results = sorted(results, key=lambda x: x[0], reverse=False)
            gamma = results[0][1]
            parameter = results[0][-len(parameter):]
            print("newParameter: " + str(parameter) + " with error: " + str(results[0][0]))

        print(results)
        print("For the " + algorithm.get_name() + " the best Parameter results are: " + str(
            results[0]) + " with randomtype: " + str(random_type))

        optimized_parameter = np.append(results[0], random_type)

        pickle.dump(optimized_parameter, open(
            "saves/" + file_to_load + algorithm.get_name() + "_Datasetsize" + str(
                len(data)) + "optimizizedParameter_notRandom",
            "wb"))

    @staticmethod
    def gaussian_mutation(parameter, gamma):
        parameter = parameter + gamma * np.random.randn(len(parameter))
        return abs(np.asarray(parameter))
