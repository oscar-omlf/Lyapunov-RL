from abc import ABC, abstractmethod

class AbstractTrainer(ABC):
    @abstractmethod
    def train(self):
        """
        Perform a training update. Must be implemented by any concrete trainer.
        """
        pass
