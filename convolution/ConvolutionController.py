from convolution.ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from support_classes.Loader import Loader


class ConvolutionController:

    @staticmethod
    def generate_cnn_data(dataset, dataset_size):
        # creates and runs a cnn. the predicted test-data is saved under /saves as a pickle-file

        data, label, color_channel, x_train, y_train, x_test, y_test, input_shape, number_of_classes, name = Loader.get_convolutional_data(
            dataset)

        cnn = ConvolutionalNeuralNetwork(color_channel, x_train, y_train, x_test, y_test, input_shape,
                                         number_of_classes, name)

        cnn.run()

        cnn.predict_and_save_layer_output_of_data(data, label, dataset_size)
