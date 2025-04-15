import os
import threading
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt

from .welford import Welford


class MetricsTracker:
    """
    Thread-safe object for recording per-episode metrics across multiple runs.
    For each agent and each episode index, a separate Welford aggregator is maintained.
    """
    def __init__(self):
        self._lock = threading.RLock()
        # For each agent, for each episode index, we store a Welford aggregator.
        self._returns = defaultdict(lambda: defaultdict(Welford))
        self._actor_losses = defaultdict(lambda: defaultdict(Welford))
        self._critic_losses = defaultdict(lambda: defaultdict(Welford))

    def add_run_returns(self, agent_id: str, returns: list) -> None:
        """
        Record a complete run of returns (a list, one per episode) for the given agent.
        """
        with self._lock:
            for ep_idx, ret in enumerate(returns):
                self._returns[agent_id][ep_idx].update_aggr(ret)

    def add_run_actor_losses(self, agent_id: str, actor_losses: list) -> None:
        """
        Record a complete run of actor losses (a list, one per episode) for the given agent.
        """
        with self._lock:
            for ep_idx, loss in enumerate(actor_losses):
                self._actor_losses[agent_id][ep_idx].update_aggr(loss)

    def add_run_critic_losses(self, agent_id: str, critic_losses: list) -> None:
        """
        Record a complete run of critic losses (a list, one per episode) for the given agent.
        """
        with self._lock:
            for ep_idx, loss in enumerate(critic_losses):
                self._critic_losses[agent_id][ep_idx].update_aggr(loss)

    def add_run_losses(self, agent_id: str, actor_losses: list, critic_losses: list) -> None:
        """
        Record a complete run of both actor and critic losses for the given agent.
        """
        self.add_run_actor_losses(agent_id, actor_losses)
        self.add_run_critic_losses(agent_id, critic_losses)

    def _get_avg_stats(self, agg_dict: dict, agent_id: str):
        """
        For a given agent and an aggregator dictionary (returns or losses),
        return a tuple (episodes, means, ses) where:
          - episodes is a sorted list of episode indices,
          - means is the list of aggregated means, and
          - ses is the list of standard errors computed as sqrt(variance/count).
        """
        episodes = sorted(agg_dict[agent_id].keys())
        means = []
        ses = []
        for ep in episodes:
            aggregator = agg_dict[agent_id][ep]
            mean, var = aggregator.get_curr_mean_variance()
            count = aggregator.count if aggregator.count > 0 else 1
            se = np.sqrt(var / count)
            means.append(mean)
            ses.append(se)
        return episodes, means, ses

    def get_avg_returns(self, agent_id: str):
        """
        Get per-episode average returns and their standard errors.
        Returns (episodes, means, ses)
        """
        with self._lock:
            return self._get_avg_stats(self._returns, agent_id)

    def get_avg_actor_losses(self, agent_id: str):
        """
        Get per-episode average actor losses and their standard errors.
        Returns (episodes, means, ses)
        """
        with self._lock:
            return self._get_avg_stats(self._actor_losses, agent_id)

    def get_avg_critic_losses(self, agent_id: str):
        """
        Get per-episode average critic losses and their standard errors.
        Returns (episodes, means, ses)
        """
        with self._lock:
            return self._get_avg_stats(self._critic_losses, agent_id)

    def plot(self) -> None:
        """
        Plot the per-episode metrics in a matplotlib figure with two subplots.
        The top subplot displays losses (actor and critic for each agent) and the
        bottom subplot displays returns. The standard error is shown as a shaded region.
        """
        with self._lock:
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

            # Plot Loss History (actor and critic)
            agents = sorted(set(list(self._actor_losses.keys()) + list(self._critic_losses.keys())))
            for agent_id in agents:
                if agent_id in self._actor_losses:
                    episodes, means, ses = self.get_avg_actor_losses(agent_id)
                    axes[0].plot(episodes, means, label=f'{agent_id}: Actor')
                    axes[0].fill_between(episodes,
                                         np.array(means) - np.array(ses),
                                         np.array(means) + np.array(ses),
                                         alpha=0.2)
                if agent_id in self._critic_losses:
                    episodes, means, ses = self.get_avg_critic_losses(agent_id)
                    axes[0].plot(episodes, means, label=f'{agent_id}: Critic')
                    axes[0].fill_between(episodes,
                                         np.array(means) - np.array(ses),
                                         np.array(means) + np.array(ses),
                                         alpha=0.2)
            axes[0].set_title('Loss History')
            axes[0].set_xlabel('Episodes')
            axes[0].set_ylabel('Average Loss')
            axes[0].legend()

            # Plot Return History
            for agent_id in sorted(self._returns.keys()):
                episodes, means, ses = self.get_avg_returns(agent_id)
                axes[1].plot(episodes, means, label=f'{agent_id} Returns')
                axes[1].fill_between(episodes,
                                     np.array(means) - np.array(ses),
                                     np.array(means) + np.array(ses),
                                     alpha=0.2)
            axes[1].set_title('Return History')
            axes[1].set_xlabel('Episodes')
            axes[1].set_ylabel('Average Return')
            axes[1].legend()

            plt.tight_layout()
            plt.show()

    def plot_split(self) -> None:
        """
        Plot the per-episode metrics in separate matplotlib figures,
        with legends inside the plot and enlarged figures for better readability.
        """
        with self._lock:
            fig_size = (15, 6)

            # Plot Loss History
            plt.figure(figsize=fig_size)
            agents = sorted(set(list(self._actor_losses.keys()) + list(self._critic_losses.keys())))
            for agent_id in agents:
                if agent_id in self._actor_losses:
                    episodes, means, ses = self.get_avg_actor_losses(agent_id)
                    plt.plot(episodes, means, label=f'{agent_id}: Actor')
                    plt.fill_between(episodes,
                                     np.array(means) - np.array(ses),
                                     np.array(means) + np.array(ses),
                                     alpha=0.2)
                if agent_id in self._critic_losses:
                    episodes, means, ses = self.get_avg_critic_losses(agent_id)
                    plt.plot(episodes, means, label=f'{agent_id}: Critic')
                    plt.fill_between(episodes,
                                     np.array(means) - np.array(ses),
                                     np.array(means) + np.array(ses),
                                     alpha=0.2)
            plt.title('Loss History')
            plt.xlabel('Episodes')
            plt.ylabel('Average Loss')
            plt.legend(loc='best', fontsize='medium')
            plt.tight_layout()
            plt.show()

            # Plot Return History
            plt.figure(figsize=fig_size)
            for agent_id in sorted(self._returns.keys()):
                episodes, means, ses = self.get_avg_returns(agent_id)
                plt.plot(episodes, means, label=f'{agent_id} Returns')
                plt.fill_between(episodes,
                                 np.array(means) - np.array(ses),
                                 np.array(means) + np.array(ses),
                                 alpha=0.2)
            plt.title('Return History')
            plt.xlabel('Episodes')
            plt.ylabel('Average Return')
            plt.legend(loc='best', fontsize='medium')
            plt.tight_layout()
            plt.show()

    def plot_top_10_agents(self) -> None:
        """
        Plot the return histories for the top 10 agents based on the final episode's average return.
        """
        with self._lock:
            # Calculate each agent's final episode average return.
            agent_final_returns = {}
            for agent_id, ep_dict in self._returns.items():
                final_ep = max(ep_dict.keys())
                aggregator = ep_dict[final_ep]
                mean, _ = aggregator.get_curr_mean_variance()
                agent_final_returns[agent_id] = mean

            # Select the top 10 agents.
            top_agents = sorted(agent_final_returns.items(), key=lambda x: x[1], reverse=True)[:10]
            top_agent_ids = [agent_id for agent_id, _ in top_agents]

            # Plot return histories for the top agents.
            plt.figure(figsize=(15, 6))
            for agent_id in top_agent_ids:
                episodes, means, ses = self.get_avg_returns(agent_id)
                plt.plot(episodes, means, label=f'{agent_id} Returns')
                plt.fill_between(episodes,
                                 np.array(means) - np.array(ses),
                                 np.array(means) + np.array(ses),
                                 alpha=0.2)
            plt.title('Top 10 Agents Return History')
            plt.xlabel('Episodes')
            plt.ylabel('Average Return')
            plt.legend(loc='best', fontsize='medium')
            plt.tight_layout()
            plt.show()

    def save_top10_plots(self, folder: str = "plots") -> None:
        """
        Save the top 10 agents' return history and combined loss (actor and critic)
        history plots as PNG files in the specified folder. The top 10 agents are selected
        based on the final episode's average return.
        """
        os.makedirs(folder, exist_ok=True)
        with self._lock:
            # Determine top 10 agents based on final return.
            agent_final_returns = {}
            for agent_id, ep_dict in self._returns.items():
                final_ep = max(ep_dict.keys())
                aggregator = ep_dict[final_ep]
                mean, _ = aggregator.get_curr_mean_variance()
                agent_final_returns[agent_id] = mean

            top_agents = sorted(agent_final_returns.items(), key=lambda x: x[1], reverse=True)[:10]
            top_agent_ids = [agent_id for agent_id, _ in top_agents]

            # Save Top 10 Returns Plot.
            fig_returns, ax_returns = plt.subplots(figsize=(15, 6))
            for agent_id in top_agent_ids:
                episodes, means, ses = self.get_avg_returns(agent_id)
                ax_returns.plot(episodes, means, label=f'{agent_id} Returns')
                ax_returns.fill_between(episodes,
                                        np.array(means) - np.array(ses),
                                        np.array(means) + np.array(ses),
                                        alpha=0.2)
            ax_returns.set_title('Top 10 Agents Return History')
            ax_returns.set_xlabel('Episodes')
            ax_returns.set_ylabel('Average Return')
            ax_returns.legend(loc='best', fontsize='medium')
            fig_returns.tight_layout()
            fig_returns.savefig(os.path.join(folder, "top10_returns.png"))
            plt.close(fig_returns)

            # Save Combined Top 10 Losses Plot (Actor and Critic together).
            fig_losses, ax_losses = plt.subplots(figsize=(15, 6))
            for agent_id in top_agent_ids:
                # Plot Actor Losses.
                episodes_actor, means_actor, ses_actor = self.get_avg_actor_losses(agent_id)
                ax_losses.plot(episodes_actor, means_actor, label=f'{agent_id} Actor Loss')
                ax_losses.fill_between(episodes_actor,
                                         np.array(means_actor) - np.array(ses_actor),
                                         np.array(means_actor) + np.array(ses_actor),
                                         alpha=0.2)
                # Plot Critic Losses.
                episodes_critic, means_critic, ses_critic = self.get_avg_critic_losses(agent_id)
                ax_losses.plot(episodes_critic, means_critic, label=f'{agent_id} Critic Loss')
                ax_losses.fill_between(episodes_critic,
                                         np.array(means_critic) - np.array(ses_critic),
                                         np.array(means_critic) + np.array(ses_critic),
                                         alpha=0.2)
            ax_losses.set_title('Top 10 Agents Loss History (Actor & Critic)')
            ax_losses.set_xlabel('Episodes')
            ax_losses.set_ylabel('Average Loss')
            ax_losses.legend(loc='best', fontsize='medium')
            fig_losses.tight_layout()
            fig_losses.savefig(os.path.join(folder, "top10_losses.png"))
            plt.close(fig_losses)

    def save_top10_losses_plot(self, folder: str = "plots") -> None:
        """
        Save the combined top 10 losses plot (actor and critic) for the top 10 agents.
        If no return data is available to rank agents, fall back to using actor loss data.
        """
        os.makedirs(folder, exist_ok=True)
        with self._lock:
            # Attempt to determine top 10 agents based on final return.
            if self._returns:
                agent_final_returns = {}
                for agent_id, ep_dict in self._returns.items():
                    final_ep = max(ep_dict.keys())
                    aggregator = ep_dict[final_ep]
                    mean, _ = aggregator.get_curr_mean_variance()
                    agent_final_returns[agent_id] = mean

                top_agents = sorted(agent_final_returns.items(), key=lambda x: x[1], reverse=True)[:10]
                top_agent_ids = [agent_id for agent_id, _ in top_agents]
            else:
                # Fallback: use all agents with loss data.
                top_agent_ids = list(self._actor_losses.keys())

            # Create and save the Combined Losses Plot (Actor and Critic together).
            fig_losses, ax_losses = plt.subplots(figsize=(15, 6))
            for agent_id in top_agent_ids:
                if agent_id in self._actor_losses:
                    episodes_actor, means_actor, ses_actor = self.get_avg_actor_losses(agent_id)
                    ax_losses.plot(episodes_actor, means_actor, label=f'{agent_id} Actor Loss')
                    ax_losses.fill_between(episodes_actor,
                                        np.array(means_actor) - np.array(ses_actor),
                                        np.array(means_actor) + np.array(ses_actor),
                                        alpha=0.2)
                if agent_id in self._critic_losses:
                    episodes_critic, means_critic, ses_critic = self.get_avg_critic_losses(agent_id)
                    ax_losses.plot(episodes_critic, means_critic, label=f'{agent_id} Critic Loss')
                    ax_losses.fill_between(episodes_critic,
                                        np.array(means_critic) - np.array(ses_critic),
                                        np.array(means_critic) + np.array(ses_critic),
                                        alpha=0.2)
            ax_losses.set_title('Top 10 Agents Loss History (Actor & Critic)')
            ax_losses.set_xlabel('Episodes')
            ax_losses.set_ylabel('Average Loss')
            ax_losses.legend(loc='best', fontsize='medium')
            fig_losses.tight_layout()
            fig_losses.savefig(os.path.join(folder, "top10_losses.png"))
            plt.close(fig_losses)
