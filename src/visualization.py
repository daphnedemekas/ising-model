"""
Visualization utilities for Ising model simulations.

This module provides reusable plotting functions and visualization tools
for analyzing simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Any, Dict
import os
from pathlib import Path

try:
    from .config import SimulationConfig, VisualizationHelper
except ImportError:
    from config import SimulationConfig, VisualizationHelper


class IsingVisualization:
    """Main visualization class for Ising model results."""

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize visualization helper.

        Args:
            output_dir: Directory for saving plots (defaults to config setting)
        """
        self.output_dir = output_dir or SimulationConfig.get_output_directory()

    def plot_activity_heatmap(self, activity_hist: np.ndarray,
                             title: str = "Activity Heatmap",
                             filename: str = None,
                             cmap: str = "viridis",
                             subsample_factor: int = None,
                             show_plot: bool = False) -> str:
        """
        Plot activity heatmap with subsampling for large networks.

        Args:
            activity_hist: Activity history array (nodes x time)
            title: Plot title
            filename: Output filename (auto-generated if None)
            cmap: Colormap name
            subsample_factor: Factor for subsampling nodes (defaults to config)
            show_plot: Whether to display the plot

        Returns:
            Path to saved figure
        """
        subsample_factor = subsample_factor or SimulationConfig.DEFAULT_SUBSAMPLE_NODES

        # Subsample nodes for better visualization
        if activity_hist.shape[0] > 100:  # Only subsample if many nodes
            subsampled_data = activity_hist[::subsample_factor, :]
        else:
            subsampled_data = activity_hist

        fig, ax = VisualizationHelper.setup_figure(1, 1, figsize=(12, 6))
        ax = ax[0]

        im = ax.imshow(subsampled_data, aspect="auto", interpolation="none", cmap=cmap)

        VisualizationHelper.apply_common_styling(
            ax, title=title, xlabel="Time Steps", ylabel="Node Index (subsampled)"
        )

        plt.colorbar(im, ax=ax, label="Activity")
        plt.tight_layout()

        if filename is None:
            filename = f"activity_heatmap_{cmap}.png"
        output_path = os.path.join(self.output_dir, filename)

        VisualizationHelper.save_figure(fig, output_path)

        if show_plot:
            plt.show()
        else:
            VisualizationHelper.close_figure(fig)

        return output_path

    def plot_dual_heatmap(self, phi_hist: np.ndarray, spin_hist: np.ndarray,
                         filename: str = "dual_heatmap.png",
                         show_plot: bool = False) -> str:
        """
        Plot side-by-side heatmaps of phi and spin histories.

        Args:
            phi_hist: Posterior belief history
            spin_hist: Spin state history
            filename: Output filename
            show_plot: Whether to display the plot

        Returns:
            Path to saved figure
        """
        fig, axes = VisualizationHelper.setup_figure(1, 2, figsize=(15, 6))

        # Subsample for better visualization
        subsample = SimulationConfig.DEFAULT_SUBSAMPLE_NODES
        if phi_hist.shape[0] > 100:
            phi_sub = phi_hist[::subsample, :]
            spin_sub = spin_hist[::subsample, :]
        else:
            phi_sub = phi_hist
            spin_sub = spin_hist

        # Phi heatmap
        im1 = axes[0].imshow(phi_sub, aspect="auto", interpolation="none", cmap="viridis")
        VisualizationHelper.apply_common_styling(
            axes[0], title="Posterior Beliefs (φ)", xlabel="Time Steps", ylabel="Node Index"
        )
        plt.colorbar(im1, ax=axes[0], label="φ value")

        # Spin heatmap
        im2 = axes[1].imshow(spin_sub, aspect="auto", interpolation="none", cmap="RdBu")
        VisualizationHelper.apply_common_styling(
            axes[1], title="Spin States", xlabel="Time Steps", ylabel="Node Index"
        )
        plt.colorbar(im2, ax=axes[1], label="Spin State")

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, filename)
        VisualizationHelper.save_figure(fig, output_path)

        if show_plot:
            plt.show()
        else:
            VisualizationHelper.close_figure(fig)

        return output_path

    def plot_average_dynamics(self, activity_hist: np.ndarray,
                             title: str = "Average Network Activity",
                             filename: str = "average_dynamics.png",
                             show_plot: bool = False) -> str:
        """
        Plot average activity over time.

        Args:
            activity_hist: Activity history (nodes x time)
            title: Plot title
            filename: Output filename
            show_plot: Whether to display the plot

        Returns:
            Path to saved figure
        """
        average_activity = activity_hist.mean(axis=0)
        time_steps = np.arange(len(average_activity))

        fig, ax = VisualizationHelper.setup_figure(1, 1, figsize=(10, 6))
        ax = ax[0]

        ax.plot(time_steps, average_activity, linewidth=2, color='steelblue')

        VisualizationHelper.apply_common_styling(
            ax, title=title, xlabel="Time Steps", ylabel="Average Activity"
        )

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, filename)
        VisualizationHelper.save_figure(fig, output_path)

        if show_plot:
            plt.show()
        else:
            VisualizationHelper.close_figure(fig)

        return output_path

    def plot_parameter_evolution(self, k_matrix_hist: np.ndarray,
                               focal_agents: List[int] = None,
                               observed_agents: List[int] = None,
                               spin_hist: np.ndarray = None,
                               filename: str = "parameter_evolution.png",
                               show_plot: bool = False) -> str:
        """
        Plot evolution of k-matrix parameters during learning.

        Args:
            k_matrix_hist: History of k-matrix values (nodes x nodes x time)
            focal_agents: Indices of focal agents to plot
            observed_agents: Indices of observed agents to plot
            spin_hist: Optional spin history to overlay
            filename: Output filename
            show_plot: Whether to display the plot

        Returns:
            Path to saved figure
        """
        if focal_agents is None:
            focal_agents = [0, 1]
        if observed_agents is None:
            observed_agents = [2, 3]

        n_focal = len(focal_agents)
        n_observed = len(observed_agents)

        fig, axes = VisualizationHelper.setup_figure(n_focal, n_observed, figsize=(15, 10))

        # Ensure axes is 2D
        if n_focal == 1 and n_observed == 1:
            axes = [[axes[0]]]
        elif n_focal == 1:
            axes = [axes]
        elif n_observed == 1:
            axes = [[ax] for ax in axes]

        time_steps = np.arange(k_matrix_hist.shape[2])

        for i, agent_i in enumerate(focal_agents):
            for j, agent_j in enumerate(observed_agents):
                ax = axes[i][j]

                # Plot k-matrix evolution
                k_values = k_matrix_hist[agent_i, agent_j, :]
                ax.plot(time_steps, k_values, label=f'K[{agent_i},{agent_j}]',
                       linewidth=2, color='steelblue')

                # Optionally overlay spin history
                if spin_hist is not None:
                    spin_values = 2 * spin_hist[agent_j, :]
                    ax.plot(time_steps, spin_values, label=f'2×Spin[{agent_j}]',
                           alpha=0.7, color='orange', linestyle='--')

                VisualizationHelper.apply_common_styling(
                    ax, title=f'Agent {agent_i} → Agent {agent_j}',
                    xlabel="Time Steps", ylabel="Parameter Value"
                )
                ax.legend()

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, filename)
        VisualizationHelper.save_figure(fig, output_path)

        if show_plot:
            plt.show()
        else:
            VisualizationHelper.close_figure(fig)

        return output_path

    def plot_regime_comparison(self, results_dict: Dict[str, Dict],
                              metric: str = 'average_activity',
                              filename: str = "regime_comparison.png",
                              show_plot: bool = False) -> str:
        """
        Plot comparison across different parameter regimes.

        Args:
            results_dict: Dictionary with regime names as keys and results as values
            metric: Metric to plot ('average_activity', 'vfe', etc.)
            filename: Output filename
            show_plot: Whether to display the plot

        Returns:
            Path to saved figure
        """
        n_regimes = len(results_dict)
        fig, axes = VisualizationHelper.setup_figure(n_regimes, 1, figsize=(12, 4*n_regimes))

        for i, (regime_name, data) in enumerate(results_dict.items()):
            ax = axes[i] if n_regimes > 1 else axes[0]

            if metric == 'average_activity' and 'phi_hist' in data:
                activity = data['phi_hist'].mean(axis=0)
                time_steps = np.arange(len(activity))
                ax.plot(time_steps, activity, linewidth=2)
                ax.set_ylabel("Average Activity")

            elif metric == 'heatmap' and 'phi_hist' in data:
                subsample = SimulationConfig.DEFAULT_SUBSAMPLE_NODES
                phi_data = data['phi_hist']
                if phi_data.shape[0] > 100:
                    phi_data = phi_data[::subsample, :]
                im = ax.imshow(phi_data, aspect="auto", interpolation="none", cmap="viridis")
                plt.colorbar(im, ax=ax)
                ax.set_ylabel("Node Index")

            VisualizationHelper.apply_common_styling(
                ax, title=f"Regime: {regime_name}", xlabel="Time Steps"
            )

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, filename)
        VisualizationHelper.save_figure(fig, output_path)

        if show_plot:
            plt.show()
        else:
            VisualizationHelper.close_figure(fig)

        return output_path

    def plot_vfe_analysis(self, po_vec: np.ndarray,
                         vfe_trials: np.ndarray,
                         kld_trials: np.ndarray,
                         accur_trials: np.ndarray,
                         filename: str = "vfe_analysis.png",
                         show_plot: bool = False) -> str:
        """
        Plot variational free energy analysis results.

        Args:
            po_vec: Parameter values
            vfe_trials: VFE results across trials
            kld_trials: Complexity results across trials
            accur_trials: Accuracy results across trials
            filename: Output filename
            show_plot: Whether to display the plot

        Returns:
            Path to saved figure
        """
        fig, axes = VisualizationHelper.setup_figure(2, 1, figsize=(12, 10))

        # VFE plot
        std_val = 1.96 * vfe_trials.std(axis=0)
        mean_val = vfe_trials.mean(axis=0)
        axes[0].fill_between(po_vec, mean_val + std_val, mean_val - std_val,
                            alpha=0.3, color='steelblue')
        axes[0].plot(po_vec, mean_val, label="VFE", linewidth=2, color='steelblue')

        VisualizationHelper.apply_common_styling(
            axes[0], title="Variational Free Energy",
            ylabel="VFE (nats)"
        )
        axes[0].legend()

        # Components plot
        std_kld = 1.96 * kld_trials.std(axis=0)
        mean_kld = kld_trials.mean(axis=0)
        axes[1].fill_between(po_vec, mean_kld + std_kld, mean_kld - std_kld,
                            alpha=0.3, color='red')
        axes[1].plot(po_vec, mean_kld, label="Complexity", linewidth=2, color='red')

        std_acc = 1.96 * accur_trials.std(axis=0)
        mean_acc = -accur_trials.mean(axis=0)
        axes[1].fill_between(po_vec, mean_acc + std_acc, mean_acc - std_acc,
                            alpha=0.3, color='green')
        axes[1].plot(po_vec, mean_acc, label="(-ve) Accuracy", linewidth=2, color='green')

        VisualizationHelper.apply_common_styling(
            axes[1], title="VFE Components",
            xlabel="Likelihood parameter (p_O)", ylabel="Average nats"
        )
        axes[1].legend()

        for ax in axes:
            ax.set_xlim(po_vec[0], po_vec[-1])

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, filename)
        VisualizationHelper.save_figure(fig, output_path)

        if show_plot:
            plt.show()
        else:
            VisualizationHelper.close_figure(fig)

        return output_path


def quick_plot_activity(activity_hist: np.ndarray, title: str = "Activity",
                       output_dir: str = None, show_plot: bool = True) -> str:
    """
    Quick utility function for plotting activity without class instantiation.

    Args:
        activity_hist: Activity history array
        title: Plot title
        output_dir: Output directory
        show_plot: Whether to display the plot

    Returns:
        Path to saved figure
    """
    viz = IsingVisualization(output_dir)
    return viz.plot_activity_heatmap(activity_hist, title=title, show_plot=show_plot)


def quick_plot_comparison(phi_hist: np.ndarray, spin_hist: np.ndarray,
                         output_dir: str = None, show_plot: bool = True) -> str:
    """
    Quick utility function for plotting phi/spin comparison.

    Args:
        phi_hist: Posterior belief history
        spin_hist: Spin state history
        output_dir: Output directory
        show_plot: Whether to display the plot

    Returns:
        Path to saved figure
    """
    viz = IsingVisualization(output_dir)
    return viz.plot_dual_heatmap(phi_hist, spin_hist, show_plot=show_plot)
