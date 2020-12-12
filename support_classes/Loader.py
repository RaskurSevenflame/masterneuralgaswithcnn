import pickle
from pathlib import Path
import numpy as np
from support_classes.Normalizer import Normalizer
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow import keras


class Loader:

    """
        Loads all the data and datasets
    """

    def load_dataset(self, dataset_size, file_to_load):
        file_to_load = file_to_load + str(dataset_size)
        dataset = self.load(file_to_load)
        dataset = np.array(dataset)
        if dataset is None:
            print("Error while loading the dataset for the Self-Organizing Maps")
            exit()
        label = dataset[:, 0:1]
        data = dataset[:, 1:]
        return data, label

    def load_neuron_informations(self, data_size, file_to_load, algortihm, optimized):
        if optimized:
            file_name = file_to_load + algortihm.get_name() + "NeuronWeights_Datasetsize" + str(data_size) + "optimized"
        else:
            file_name = file_to_load + algortihm.get_name() + "NeuronWeights_Datasetsize" + str(data_size)
        weights = self.load(file_name)
        if weights is None:
            print("Weights are None")
            exit()
        
        return weights

    def load_parameter(self, file_to_load, organizer, data):
        parameter_to_load = file_to_load + organizer.get_name() + "_Datasetsize" + str(
            len(data)) + "optimizizedParameter_notRandom"
        parameter = self.load(parameter_to_load)
        if parameter is None:
            print("Parameter are None")
            exit()
        return parameter

    @staticmethod
    def load(file_name):
        root = Path(".")
        file_path = root / "saves" / file_name
        with open(file_path, 'rb') as handle:
            result = pickle.load(handle)
        if result is None:
            print("Loader-Result are None")
        return result

    @staticmethod
    def get_convolutional_data(dataset):
        # returns the data and labels of the given dataset, after normalizing them
        # furthermore it saves the train and test data for further usage in this class

        x_train = []
        y_train = []
        x_test = []
        y_test = []
        pictureData = False
        color_channel = 0
        name = "none"

        if dataset is None:
            x_train.append([1, 0, 0])  # red
            y_train.append("red")
            x_train.append([0, 1, 0])  # green
            y_train.append("green")
            x_train.append([0, 0.5, 0.25])  # darkgreen
            y_train.append("darkgreen")
            x_train.append([0, 0, 1])  # blue
            y_train.append("blue")
            x_train.append([0, 0, 0.5])  # darkblue
            y_train.append("darkblue")
            x_train.append([1, 1, 0.2])  # yellow
            y_train.append("yellow")
            x_train.append([1, 0.4, 0.25])  # orange
            y_train.append("orange")
            x_train.append([1, 0, 1])  # purple
            y_train.append("purple")

            x_test.append([0.95, 0, 0])
            y_test.append("red")
            x_test.append([0.95, 0, 0.9])
            y_test.append("purple")
            x_test.append([0, 0, 0.9])
            y_test.append("blue")

            number_of_classes = 8

            input_shape = (3,)

            name = "RandomPoints"

            color_channel = 3
            label = np.append(y_train, y_test, axis=0)

        else:
            number_of_classes = 0
            pictureData = True

            if dataset == "mnist":
                (x_train, y_train), (x_test, y_test) = mnist.load_data()
                number_of_classes = 10
                color_channel = 1
                name = "mnist"

            if dataset == "cifar10":
                (x_train, y_train), (x_test, y_test) = cifar10.load_data()
                number_of_classes = 10
                color_channel = 3
                name = "cifar10"

            x_train = x_train.reshape(x_train.shape[0], x_train[1].shape[0], x_train[1].shape[1], color_channel)
            x_test = x_test.reshape(x_test.shape[0], x_test[1].shape[0], x_test[1].shape[1], color_channel)
            input_shape = x_train[0].shape
            label = np.append(y_train, y_test, axis=0)
            y_train = keras.utils.to_categorical(y_train, number_of_classes)
            y_test = keras.utils.to_categorical(y_test, number_of_classes)

        if pictureData:
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train = (x_train - 0) / (255 - 0)
            x_test = (x_test - 0) / (255 - 0)
        else:
            Normalizer.normalize([x_train, x_test])

        data = np.append(x_train, x_test, axis=0)

        return data, label, color_channel, x_train, y_train, x_test, y_test, input_shape, number_of_classes, name