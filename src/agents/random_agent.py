from agents.abstract_agent import AbstractAgent

class RandomAgent(AbstractAgent):
    def __init__(self, config):
        super().__init__(config)

    def policy(self, state):
        """
        Return a random action sampled from the environment's action space.
        Assumes that the config dictionary has been updated with the key "action_space".
        """
        return self.config["action_space"].sample()

    def update(self):
        """
        Random agent does not learn.
        """
        return None

    def save(self, filepath):
        """
        Random agent has no parameters to save.
        """
        pass

    def load(self, filepath):
        """
        Random agent has no parameters to load.
        """
        pass
