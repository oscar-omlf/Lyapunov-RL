from agents.random_agent import RandomAgent

class AgentFactory:
    @staticmethod
    def create_agent(config, env):
        """
        Create an agent instance based on the provided configuration.
        The config dictionary is updated with the environment's state_space and action_space.
        """
        config["state_space"] = env.observation_space
        config["action_space"] = env.action_space

        agent_str = config.get("agent_str", "RANDOM").upper()

        if agent_str == "RANDOM":
            return RandomAgent(config)
        elif agent_str == "ACTOR-CRITIC":
            pass
        elif agent_str == "LQR":
            pass
        elif agent_str == "LYAPUNOV":
            pass
        else:
            raise ValueError(f"Unknown agent type: {agent_str}")
