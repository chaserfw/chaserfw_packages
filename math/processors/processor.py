from abc import ABC, abstractmethod
##################################################################################
class CorrelationProcessor(ABC):
    def __init__(self, *args):
        """
        docstring
        """
        pass

    @abstractmethod
    def __call__(self, *args):
        """
        docstring
        """
        raise NotImplementedError

    def run(self, *args):
        self.__call__(args)

##################################################################################
class SboxProcessor(CorrelationProcessor):
    def __init__(self, byte1, byte2,*args):
        """
        docstring
        """
        super(CorrelationProcessor, self).__init__(*args)
        self.__byte1 = byte1
        self.__byte2 = byte2

    def __call__(self, *args):
        """
        docstring
        """
        