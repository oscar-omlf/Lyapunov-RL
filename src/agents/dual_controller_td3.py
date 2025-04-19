import numpy as np

from agents.abstract_agent import AbstractAgent
from agents.td3_agent import TD3Agent
from agents.lqr_agent import LQRAgent
from util.domain_estimation import estimate_LQR_domain_of_attraction


class DualControllerTD3(AbstractAgent):
    def __init__(self, config):
        super().__init__(config)

        self.td3_agent = TD3Agent(config)
        self.lqr_agent = LQRAgent(config)

        self.beta = config.get("beta", 0.5)
        self.s = np.arctanh(self.beta)

        self.x_star = 0 # write np or torch dobule zero here. Which one should it be?
        self.c_star = estimate_LQR_domain_of_attraction()

    def compute_blending_factor(self, state):
        # Implement this function
        pass

    def add_transition(self, transition):
        """
        Add a transition to the replay buffer (delegated to the TD3 agent).
        """
        self.td3_agent.add_transition(transition)

    def update(self) -> tuple:
        """
        Update the agent (we update the TD3 network only, as the LQR component is fixed).
        """
        loss = self.td3_agent.update()
        return loss

    def policy(self, state):
        """
        Return the blended action: LQR + h(x)*(TD3 - LQR).
        Assumes state is a NumPy array.
        """
        action_td3 = self.td3_agent.policy(state)
        action_lqr = self.lqr_agent.policy(state)
        
        blending_factor = self.h1_func(state)
        
        dual_action = action_lqr + blending_factor * (action_td3 - action_lqr)
        return dual_action
    
    def save(self, file_path: str = './') -> None:
        """
        Save both TD3 and LQR models.
        """
        self.td3_agent.save(file_path)
        self.lqr_agent.save(file_path)

    def load(self, file_path: str = './') -> None:
        """
        Load both TD3 and LQR models.
        """
        self.td3_agent.load(file_path)
        self.lqr_agent.load(file_path)
