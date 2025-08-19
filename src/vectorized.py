"""
Vectorized Ising-like Network Dynamics Simulation

This module provides a vectorized implementation of Ising-like network dynamics
with learning capabilities and comprehensive visualization tools.
"""

import os
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, List
import warnings
try:
    from simulation.simulation import SimulationVectorized
except ImportError:
    try:
        from .simulation.simulation import SimulationVectorized
    except ImportError:
        SimulationVectorized = None

# Suppress potential import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class VectorizedIsingSimulator:
    """
    A class for running vectorized Ising-like simulations with learning capabilities.
    """

    def __init__(self, n_nodes: int = 512, connection_prob: float = 0.01,
                 output_dir: Optional[str] = None):
        """
        Initialize the vectorized simulator.

        Args:
            n_nodes: Number of nodes in the network
            connection_prob: Connection probability for network generation
            output_dir: Directory for saving output files (defaults to tmp_out)
        """
        self.n_nodes = n_nodes
        self.connection_prob = connection_prob
        self.output_dir = output_dir or self._get_default_output_dir()
        self.graph = None
        self.k_matrix = None
        self.ps_vec = None

    def _get_default_output_dir(self) -> str:
        """Get default output directory relative to this file."""
        output_dir = Path(__file__).parent.parent / "tmp_out"
        output_dir.mkdir(exist_ok=True)
        return str(output_dir)

    def create_network(self, network_type: str = 'erdos_renyi', **kwargs) -> nx.Graph:
        """
        Create a network of specified type.

        Args:
            network_type: Type of network ('erdos_renyi', 'ring_of_cliques')
            **kwargs: Additional parameters for network creation

        Returns:
            NetworkX graph object
        """
        if network_type == 'erdos_renyi':
            self.graph = nx.fast_gnp_random_graph(self.n_nodes, self.connection_prob)

        elif network_type == 'ring_of_cliques':
            num_cliques = kwargs.get('num_cliques', 16)
            nodes_per_clique = kwargs.get('nodes_per_clique', 32)
            self.graph = nx.ring_of_cliques(num_cliques, nodes_per_clique)
            self.n_nodes = self.graph.number_of_nodes()  # Update actual number

        else:
            raise ValueError(f"Unknown network type: {network_type}")

        return self.graph

    def initialize_parameters(self, k_range: Tuple[float, float] = (0.9, 1.1),
                            omega_range: Tuple[float, float] = (0.5, 0.7),
                            ps_value: float = 0.5,
                            init_scale: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize simulation parameters.

        Args:
            k_range: Range for k_matrix values
            omega_range: Range for omega values
            ps_value: Prior probability value
            init_scale: Initialization scale for learning simulations

        Returns:
            Tuple of (k_matrix, ps_vec)
        """
        if self.graph is None:
            raise ValueError("Network not created. Call create_network() first.")

        self.k_matrix = np.random.uniform(*k_range, size=(self.n_nodes, self.n_nodes))
        self.ps_vec = ps_value * np.ones(self.n_nodes)

        return self.k_matrix, self.ps_vec

    def run_basic_simulation(self, time_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run a basic vectorized simulation.

        Args:
            time_steps: Number of simulation time steps

        Returns:
            Tuple of (phi_hist, spin_hist)
        """
        if self.graph is None or self.k_matrix is None:
            raise ValueError("Network and parameters not initialized.")

        if SimulationVectorized is None:
            raise ImportError("SimulationVectorized class not available.")

        sim = SimulationVectorized(
            G=self.graph,
            k_matrix=self.k_matrix,
            p_s_vec=self.ps_vec
        )

        phi_hist, spin_hist = sim.run(time_steps)
        return phi_hist, spin_hist

    def run_learning_simulation(self, time_steps: int = 1000,
                              init_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run a simulation with learning dynamics.

        Args:
            time_steps: Number of simulation time steps
            init_scale: Initialization scale

        Returns:
            Tuple of (phi_hist, spin_hist, k_matrix_hist)
        """
        if self.graph is None or self.k_matrix is None:
            raise ValueError("Network and parameters not initialized.")

        if SimulationVectorized is None:
            raise ImportError("SimulationVectorized class not available.")

        sim = SimulationVectorized(
            G=self.graph,
            k_matrix=self.k_matrix,
            p_s_vec=self.ps_vec,
            init_scale=init_scale
        )

        phi_hist, spin_hist, k_matrix_hist = sim.run_learning(time_steps)
        return phi_hist, spin_hist, k_matrix_hist

    def plot_basic_results(self, phi_hist: np.ndarray, spin_hist: np.ndarray,
                          filename: str = "basic_simulation.png",
                          show_plot: bool = False) -> None:
        """
        Plot and save basic simulation results.

        Args:
            phi_hist: History of posterior beliefs
            spin_hist: History of spin states
            filename: Output filename
            show_plot: Whether to display the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

        axes[0].imshow(phi_hist, aspect="auto", interpolation="none", cmap="viridis")
        axes[0].set_title("Posterior Beliefs (φ) Over Time", fontsize=14)
        axes[0].set_xlabel("Time Steps")
        axes[0].set_ylabel("Node Index")

        axes[1].imshow(spin_hist, aspect="auto", interpolation="none", cmap="RdBu")
        axes[1].set_title("Spin States Over Time", fontsize=14)
        axes[1].set_xlabel("Time Steps")
        axes[1].set_ylabel("Node Index")

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Basic simulation plot saved to: {output_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_learning_results(self, phi_hist: np.ndarray, spin_hist: np.ndarray,
                            k_matrix_hist: np.ndarray,
                            focal_agents: List[int] = None,
                            observed_agents: List[int] = None,
                            filename_base: str = "learning_simulation",
                            show_plots: bool = False) -> None:
        """
        Plot and save learning simulation results.

        Args:
            phi_hist: History of posterior beliefs
            spin_hist: History of spin states
            k_matrix_hist: History of k_matrix evolution
            focal_agents: List of focal agent indices
            observed_agents: List of observed agent indices
            filename_base: Base filename for outputs
            show_plots: Whether to display the plots
        """
        if focal_agents is None:
            focal_agents = [0, 1]
        if observed_agents is None:
            observed_agents = [3, 4]

        # Plot k-matrix evolution
        fig, axes = plt.subplots(
            len(focal_agents), len(observed_agents),
            figsize=(12, 8),
            squeeze=False
        )

        for i, agent_i in enumerate(focal_agents):
            for j, agent_j in enumerate(observed_agents):
                axes[i, j].plot(
                    k_matrix_hist[agent_i, agent_j, :],
                    label=f'K[{agent_i},{agent_j}]',
                    linewidth=2
                )
                axes[i, j].plot(
                    2 * spin_hist[agent_j, :],
                    label=f'2×Spin[{agent_j}]',
                    alpha=0.7
                )
                axes[i, j].set_title(f'Agent {agent_i} → Agent {agent_j}')
                axes[i, j].legend()
                axes[i, j].grid(True, alpha=0.3)

        plt.tight_layout()
        k_output_path = os.path.join(self.output_dir, f"{filename_base}_k_evolution.png")
        plt.savefig(k_output_path, dpi=300, bbox_inches="tight")
        print(f"K-matrix evolution plot saved to: {k_output_path}")

        if show_plots:
            plt.show()
        else:
            plt.close()

        # Plot basic activity patterns
        self.plot_basic_results(phi_hist, spin_hist,
                               f"{filename_base}_activity.png", show_plots)

    def run_complete_demo(self, network_type: str = 'erdos_renyi',
                         time_steps: int = 1000,
                         include_learning: bool = True,
                         show_plots: bool = False) -> dict:
        """
        Run a complete demonstration with both basic and learning simulations.

        Args:
            network_type: Type of network to create
            time_steps: Number of simulation time steps
            include_learning: Whether to run learning simulation
            show_plots: Whether to display plots

        Returns:
            Dictionary containing simulation results
        """
        print(f"Running complete demo with {network_type} network...")

        # Setup
        self.create_network(network_type)
        self.initialize_parameters()

        results = {}

        # Basic simulation
        print("Running basic simulation...")
        phi_hist, spin_hist = self.run_basic_simulation(time_steps)
        results['basic'] = {'phi_hist': phi_hist, 'spin_hist': spin_hist}

        self.plot_basic_results(phi_hist, spin_hist,
                               f"demo_{network_type}_basic.png", show_plots)

        # Learning simulation
        if include_learning:
            print("Running learning simulation...")
            phi_hist_learn, spin_hist_learn, k_matrix_hist = self.run_learning_simulation(time_steps)
            results['learning'] = {
                'phi_hist': phi_hist_learn,
                'spin_hist': spin_hist_learn,
                'k_matrix_hist': k_matrix_hist
            }

            self.plot_learning_results(
                phi_hist_learn, spin_hist_learn, k_matrix_hist,
                filename_base=f"demo_{network_type}_learning",
                show_plots=show_plots
            )

        print("Demo complete!")
        return results


def main():
    """
    Main function demonstrating the usage of the VectorizedIsingSimulator.
    """
    # Initialize simulator
    simulator = VectorizedIsingSimulator(n_nodes=512, connection_prob=0.01)

    # Run complete demo with Erdős-Rényi network
    print("=== Erdős-Rényi Network Demo ===")
    er_results = simulator.run_complete_demo(
        network_type='erdos_renyi',
        time_steps=1000,
        include_learning=True,
        show_plots=False
    )

    # Run demo with ring of cliques
    print("\n=== Ring of Cliques Demo ===")
    clique_results = simulator.run_complete_demo(
        network_type='ring_of_cliques',
        time_steps=1000,
        include_learning=True,
        show_plots=False
    )

    print("\nAll simulations completed successfully!")
    return er_results, clique_results


if __name__ == "__main__":
    main()
