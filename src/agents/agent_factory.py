import gymnasium as gym

from agents.random_agent import RandomAgent
from agents.actor_critic_agent import ActorCriticAgent
from agents.lqr_agent import LQRAgent
from agents.lyapunov_agent import LyapunovAgent
from agents.td3_agent import TD3Agent
from agents.las_lyac_agent import LAS_LyACAgent


class AgentFactory:
    @staticmethod
    def create_agent(config, env = None):
            """
            Create an agent instance based on the provided configuration.
            The config dictionary is updated with the environment's state_space and action_space.
            """

            if env is not None:
                config["state_space"] = env.observation_space
                config["action_space"] = env.action_space

            agent_str = config.get("agent_str", "RANDOM").upper()

            if agent_str == "RANDOM":
                return RandomAgent(config)
            elif agent_str == "AC":
                print('Creating ActorCriticAgent')
                return ActorCriticAgent(config)
            elif agent_str == "LQR":
                print('Creating LQRAgent')
                return LQRAgent(config)
            elif agent_str == "LYAPUNOV-AC":
                print('Creating LyapunovAgent')
                return LyapunovAgent(config)
            elif agent_str == 'TD3':
                 return TD3Agent(config)
            elif agent_str == 'LAS-LAC':
                 return LAS_LyACAgent(config)
            else:
                raise ValueError(f"Unknown agent type: {agent_str}")
