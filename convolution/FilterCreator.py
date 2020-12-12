
class FilterCreator:
    """
       Filterclass should / might be extended later, as more filters are tested
    """
    @staticmethod
    def create_filters():
        # for simplicity the values of the paper of kramer and elend were copied,
        # as they lead to the best value evaluation regarding their work

        filters = []
        kernels = []
        strides = []

        filters.append(32)
        kernels.append(3)
        strides.append(1)

        filters.append(64)
        kernels.append(3)
        strides.append(1)

        filters.append(128)
        kernels.append(2)
        strides.append(1)

        pooling = []
        pooling.append(2)
        pooling.append(2)
        pooling.append(2)

        return filters, kernels, strides, pooling
