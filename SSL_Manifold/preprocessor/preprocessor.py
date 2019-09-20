import abc
import numpy as np


class Preprocessor:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def preprocess(self, data):
        pass

    @abc.abstractmethod
    def preprocess_all(self, data):
        pass


class Preprocessors:

    def __init__(self):
        self.preprocessor = []

    def add(self, preprocessor):
        self.preprocessor.append(preprocessor)

    def apply(self, data):

        for processor in self.preprocessor:
            data = processor.preprocess_all(data)

        return np.asarray(data)

    def preprocess(self, data):

        for processor in self.preprocessor:
            data = processor.preprocess(data)

        return data