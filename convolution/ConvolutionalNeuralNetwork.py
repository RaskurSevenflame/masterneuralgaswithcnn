import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout
import keras.backend as KerasBackend
import pickle
from convolution.FilterCreator import FilterCreator
from support_classes.Normalizer import Normalizer


class ConvolutionalNeuralNetwork:

    def __init__(self, color_channel, x_train, y_train, x_test, y_test, input_shape, number_of_classes, name):
        self.color_channel = color_channel
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.input_shape = input_shape
        self.number_of_classes = number_of_classes
        self.name = name

    def run(self):

        # epochs, number_of_layers and batch_size are fixed to this values to ensure an ongoing fair comparison
        # the model gets fixed filters, kenels, pooling-layer and strides, which are imported from the FilterCreator

        number_of_layers = 3
        self.epochs = 10
        self.batch_size = 100

        # creates the filter for the three layers
        filter_creator = FilterCreator()
        self.filters, self.kernels, self.strides, self.pooling = filter_creator.create_filters()

        # the models_data can be accessed after each layer
        model = self.create_model(self.input_shape, number_of_layers)

        if model is not None:
            model = self.train_model(model, self.x_train, self.y_train, self.x_test, self.y_test)

        self.model = model

    def create_model(self, input_shape, number_of_layers):
        # layers of the model have fixed names for later access

        self.input_shape = input_shape

        if number_of_layers < 1:
            return None

        model = Sequential()
        model.add(Conv2D(name="Conv2D_Layer0", filters=self.filters[0], kernel_size=(self.kernels[0], self.kernels[0]),
                         strides=self.strides[0],
                         activation="relu", input_shape=input_shape))
        model.add(keras.layers.MaxPooling2D(name="Pooling_Layer0", pool_size=(self.pooling[0], self.pooling[0]),
                                            strides=self.pooling[0], padding="valid"))

        # this loop will always result in a model with 3 Conv2D layers, to ensure comparison
        for i in range(1, number_of_layers):
            model.add(Conv2D(name="Conv2D_Layer" + str(i), filters=self.filters[i],
                             kernel_size=(self.kernels[i], self.kernels[i]), strides=self.strides[i],
                             activation="relu"))
            model.add(
                keras.layers.MaxPooling2D(name="Pooling_Layer" + str(i), pool_size=(self.pooling[i], self.pooling[i]),
                                          strides=self.pooling[i], padding="valid"))

        model.add(Dropout(0.25))
        model.add(Flatten(name="Flatten"))
        model.add(Dense(self.number_of_classes, activation="softmax"))

        model.compile(optimizer="adam", loss="mse", run_eagerly=True)

        return model

    def train_model(self, model, x_train, y_train, x_test, y_test):
        # trains the model

        callbacks = []
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, callbacks=callbacks,
            validation_data=(x_test, y_test), shuffle=True
        )

        return model

    def predict_and_save_layer_output_of_data(self, data, label, dataset_size):
        # this method predicts the result for each data_point and saves the corresponding output of each pooling layer
        # instead of a reshape of the zero-layer the data is simply flattend

        zero_layer_model = Sequential()
        zero_layer_model.add(Flatten(input_shape=self.input_shape))
        zero_layer_model.compile()

        if len(self.x_test) <= dataset_size:
            data = self.x_test[:dataset_size]
            label = self.y_test[:dataset_size]

        result = None
        ready = False
        right_side = 0
        while not ready:
            left_side = right_side
            if right_side + self.batch_size * 10 < dataset_size:
                right_side = right_side + self.batch_size * 10
            else:
                right_side = dataset_size
                ready = True

            print("Working on: " + str(right_side) + " of " + str(dataset_size) + " in Layer: " + str(0) + " von 3")
            intermediate_result = zero_layer_model([data[left_side:right_side]])
            shaped = np.asarray(intermediate_result)

            if result is None:
                result = shaped
            else:
                result = np.append(result, shaped, axis=0)

        result = Normalizer.normalize(result)
        fullset = np.column_stack((label[:dataset_size], result))
        pickle.dump(fullset, open("saves/" + self.name + "with" + str(0) + "Layer" + str(dataset_size), "wb"))

        for i in range(0, 3):
            layer_output = self.model.get_layer("Pooling_Layer" + str(i)).output
            output = KerasBackend.function([self.model.input], [layer_output])

            result = None
            ready = False
            right_side = 0
            while not ready:
                left_side = right_side
                if right_side + self.batch_size * 10 < dataset_size:
                    right_side = right_side + self.batch_size * 10
                else:
                    right_side = dataset_size
                    ready = True
                print("Working on: " + str(right_side) + " of " + str(dataset_size) + " in Layer: " + str(
                    i + 1) + " von 3")
                intermediate_result = output([data[left_side:right_side]])[0]
                shaped = []
                for r in intermediate_result:
                    shaped.append(np.reshape(r, intermediate_result.shape[1] * intermediate_result.shape[2] *
                                             intermediate_result.shape[3]))

                if result is None:
                    result = shaped
                else:
                    result = np.append(result, shaped, axis=0)

            result = Normalizer.normalize(result)
            print("Layer: " + str(1 + i) + " hat eine Länge von: " + str(len(result[0])))
            fullset = np.column_stack((label[:dataset_size], result))
            pickle.dump(fullset, open("saves/" + self.name + "with" + str(i + 1) + "Layer" + str(dataset_size), "wb"))
