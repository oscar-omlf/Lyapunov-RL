from agents.abstract_agent import AbstractAgent


class RandomAgent(AbstractAgent):
    def __init__(self, config):
        super().__init__(config)
        
    def add_transition(self, transition):
        pass

    def update(self) -> None:
        pass

    def policy(self, state):
        return self.action_space.sample()

    def save(self, file_path='./') -> None:
        pass

    def load(self, file_path='./') -> None:
        pass
