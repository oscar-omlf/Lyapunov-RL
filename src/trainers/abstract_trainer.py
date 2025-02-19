from abc import ABC, abstractmethod


class Trainer(ABC):
    """
    Abstract base class for trainers responsible for performing optimization processes.
    """

    @abstractmethod
    def train(self) -> None:
        """
        Perform the optimization process or learning of parameters.
        """
        pass
