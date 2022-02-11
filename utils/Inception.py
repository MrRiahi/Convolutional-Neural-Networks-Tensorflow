

class GoogLeNet:

    def __init__(self, input_shape, classes):
        self.input_shape = input_shape
        self.classes = classes

    @staticmethod
    def _inception_block(filters, filter_size, pooling_size):
        pass