"""
Ising-like Network Dynamics Simulation

This module provides a clean implementation of Ising-like network dynamics
with variational free energy calculations and visualization capabilities.
"""

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from typing import Tuple, Optional
import warnings
# from scipy.signal import hilbert  # Not currently used

# Suppress potential import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Try to import the Simulation class
try:
    from simulation import Simulation
except ImportError:
    try:
        from ising_model.simulation import Simulation
    except ImportError:
        Simulation = None
        print("Warning: Simulation class not found. Using built-in implementation.")


class IsingNetworkSimulator:
    """
    A class for simulating Ising-like dynamics on networks with variational free energy calculations.
    """

    def __init__(self, n_nodes: int = 500, connection_prob: float = 0.01):
        """
        Initialize the simulator.

        Args:
            n_nodes: Number of nodes in the network
            connection_prob: Connection probability for Erdős-Rényi graph
        """
        self.n_nodes = n_nodes
        self.connection_prob = connection_prob
        self.graph = None
        self.adjacency_matrix = None

    def create_network(self, network_type: str = 'erdos_renyi', **kwargs) -> nx.Graph:
        """
        Create a network of specified type.

        Args:
            network_type: Type of network ('erdos_renyi', 'watts_strogatz', 'grid_2d')
            **kwargs: Additional parameters for network creation

        Returns:
            NetworkX graph object
        """
        if network_type == 'erdos_renyi':
            self.graph = nx.fast_gnp_random_graph(self.n_nodes, self.connection_prob)

        elif network_type == 'watts_strogatz':
            k = kwargs.get('k', 4)  # number of links
            p = kwargs.get('p', 0.0)  # rewiring probability
            self.graph = nx.watts_strogatz_graph(self.n_nodes, k, p)

        elif network_type == 'grid_2d':
            periodic = kwargs.get('periodic', True)
            d1 = int(np.sqrt(self.n_nodes))
            d2 = int(self.n_nodes / d1)
            self.n_nodes = d1 * d2  # Adjust to make it a perfect rectangle
            self.graph = nx.grid_2d_graph(d1, d2, periodic=periodic)

        else:
            raise ValueError(f"Unknown network type: {network_type}")

        self.adjacency_matrix = nx.to_numpy_array(self.graph)
        return self.graph

    def initialize_system(self, init_method: str = 'random_posteriors') -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize spin states and posteriors.

        Args:
            init_method: Initialization method ('random_spins', 'random_posteriors')

        Returns:
            Tuple of (initial_spins, initial_posteriors)
        """
        if init_method == 'random_spins':
            # Sample spins first, then derive posteriors
            initial_spins = np.random.randint(2, size=self.n_nodes).astype(float)
            initial_posteriors = np.abs(initial_spins - 1.0)

        elif init_method == 'random_posteriors':
            # Sample posteriors first, then derive spins by thresholding
            initial_posteriors = np.random.rand(self.n_nodes)
            initial_spins = (initial_posteriors > 0.5).astype(float)

        else:
            raise ValueError(f"Unknown initialization method: {init_method}")

        return initial_spins, initial_posteriors

    def run_simulation(self,
                      time_steps: int,
                      initial_spins: np.ndarray,
                      initial_posteriors: np.ndarray,
                      po: float,
                      ps: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the Ising-like simulation.

        Args:
            time_steps: Number of simulation time steps
            initial_spins: Initial spin configuration
            initial_posteriors: Initial posterior beliefs
            po: Observation likelihood parameter
            ps: Prior probability parameter

        Returns:
            Tuple of (phi_hist, spin_hist, kld_hist, accur_hist)
        """
        if self.adjacency_matrix is None:
            raise ValueError("Network not created. Call create_network() first.")

        # Pre-compute log probabilities for efficiency
        log_po = np.log(po)
        log_ps = np.log(ps)
        log_po_c = np.log(1 - po)
        log_ps_c = np.log(1 - ps)

        # Initialize state variables
        spin_state = initial_spins.copy()
        phi = initial_posteriors.copy()

        # Initialize history arrays
        spin_hist = np.zeros((self.n_nodes, time_steps))
        phi_hist = np.zeros((self.n_nodes, time_steps))
        kld_hist = np.zeros((self.n_nodes, time_steps))
        accur_hist = np.zeros((self.n_nodes, time_steps))

        for t in range(time_steps):
            # Calculate neighbor influences
            sum_down_spins = self.adjacency_matrix @ spin_state
            up_spins = np.abs(spin_state - 1.0)
            sum_up_spins = self.adjacency_matrix @ up_spins
            spin_diffs = sum_down_spins - sum_up_spins

            # Update posteriors using variational update rule
            x = (1.0 - (1.0 / ps)) * ((1.0 - po) / po) ** spin_diffs
            phi = 1.0 / (1.0 - x)

            # Sample new spin states
            spin_state = (phi > np.random.rand(self.n_nodes)).astype(float)

            # Store histories
            spin_hist[:, t] = spin_state.copy()
            phi_hist[:, t] = phi.copy()

            # Calculate variational free energy components
            phi_c = 1.0 - phi

            # Complexity (KL divergence)
            kld = (phi * (np.log(phi + 1e-16) - log_ps) +
                  phi_c * (np.log(phi_c + 1e-16) - log_ps_c))

            # Accuracy (expected log-likelihood)
            accur = (phi * (sum_down_spins * log_po + sum_up_spins * log_po_c) +
                    phi_c * (sum_up_spins * log_po + sum_down_spins * log_po_c))

            kld_hist[:, t] = kld
            accur_hist[:, t] = accur

        return phi_hist, spin_hist, kld_hist, accur_hist

    def plot_activity_heatmap(self, spin_hist: np.ndarray, title: str = "Spin Activity"):
        """
        Plot a heatmap of spin activity over time.

        Args:
            spin_hist: History of spin states
            title: Plot title
        """
        plt.figure(figsize=(12, 8))
        plt.imshow(spin_hist[::10, :], aspect="auto", interpolation="none", cmap="gray")
        plt.xlabel("Time", fontsize=14)
        plt.ylabel("Node (subsampled)", fontsize=14)
        plt.title(title, fontsize=16)
        plt.colorbar(label="Spin State")
        plt.tight_layout()

    def plot_regime_comparison(self, po_values: list, time_steps: int = 1000):
        """
        Plot comparison of different dynamical regimes.

        Args:
            po_values: List of observation likelihood parameters to compare
            time_steps: Number of time steps for simulation
        """
        if self.adjacency_matrix is None:
            self.create_network()

        fig, axes = plt.subplots(nrows=len(po_values), ncols=1,
                                figsize=(12, 4 * len(po_values)),
                                sharex=True)

        if len(po_values) == 1:
            axes = [axes]

        for i, po in enumerate(po_values):
            # Run simulation for this po value
            initial_spins, initial_posteriors = self.initialize_system()
            phi_hist, spin_hist, _, _ = self.run_simulation(
                time_steps, initial_spins, initial_posteriors, po
            )

            # Plot activity heatmap
            axes[i].imshow(spin_hist[::10, :], aspect="auto",
                          interpolation="none", cmap="gray")
            axes[i].set_xlim(0, time_steps)
            axes[i].set_title(f"$p_{{\\mathcal{{O}}}} = {po:.3f}$", fontsize=16)
            axes[i].tick_params(axis="both", which="major", labelsize=12)

        axes[-1].set_xlabel("Time", fontsize=14)
        axes[len(po_values)//2].set_ylabel("Node (subsampled)", fontsize=14)
        plt.tight_layout()

    def plot_average_dynamics(self, po_values: list, time_steps: int = 1000):
        """
        Plot average dynamics for different parameter values.

        Args:
            po_values: List of observation likelihood parameters
            time_steps: Number of time steps
        """
        if self.adjacency_matrix is None:
            self.create_network()

        fig, axes = plt.subplots(nrows=len(po_values), ncols=1,
                                figsize=(12, 4 * len(po_values)),
                                sharex=True)

        if len(po_values) == 1:
            axes = [axes]

        for i, po in enumerate(po_values):
            initial_spins, initial_posteriors = self.initialize_system()
            phi_hist, _, _, _ = self.run_simulation(
                time_steps, initial_spins, initial_posteriors, po
            )

            average_activity = phi_hist.mean(axis=0)
            axes[i].plot(np.arange(time_steps), average_activity)
            axes[i].set_xlim(0, time_steps)
            axes[i].set_title(f"$p_{{\\mathcal{{O}}}} = {po:.3f}$", fontsize=16)
            axes[i].tick_params(axis="both", which="major", labelsize=12)

        axes[-1].set_xlabel("Time", fontsize=14)
        axes[len(po_values)//2].set_ylabel("Average Activity", fontsize=14)
        plt.tight_layout()

    def parameter_sweep(self,
                       po_range: Tuple[float, float] = (0.5, 0.999),
                       n_points: int = 50,
                       n_trials: int = 10,
                       time_steps: int = 1000,
                       ps: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform parameter sweep over po values.

        Args:
            po_range: Range of po values (min, max)
            n_points: Number of points in parameter sweep
            n_trials: Number of trials per parameter value
            time_steps: Number of simulation time steps
            ps: Prior probability parameter

        Returns:
            Tuple of (po_vec, vfe_all_trials, kld_all_trials, accur_all_trials)
        """
        po_vec = np.linspace(po_range[0], po_range[1], n_points)
        vfe_all_trials = np.zeros((n_trials, len(po_vec)))
        kld_all_trials = np.zeros((n_trials, len(po_vec)))
        accur_all_trials = np.zeros((n_trials, len(po_vec)))

        for i, po in enumerate(po_vec):
            print(f"Processing po = {po:.3f} ({i+1}/{len(po_vec)})")

            for trial in range(n_trials):
                # Create new network for each trial
                self.create_network()
                initial_spins, initial_posteriors = self.initialize_system()

                phi_hist, spin_hist, kld_hist, accur_hist = self.run_simulation(
                    time_steps, initial_spins, initial_posteriors, po, ps
                )

                vfe_hist = kld_hist - accur_hist
                vfe_all_trials[trial, i] = vfe_hist.mean()
                kld_all_trials[trial, i] = kld_hist.mean()
                accur_all_trials[trial, i] = accur_hist.mean()

        return po_vec, vfe_all_trials, kld_all_trials, accur_all_trials

    def plot_vfe_analysis(self, po_vec: np.ndarray,
                         vfe_all_trials: np.ndarray,
                         kld_all_trials: np.ndarray,
                         accur_all_trials: np.ndarray):
        """
        Plot variational free energy analysis results.

        Args:
            po_vec: Parameter values
            vfe_all_trials: VFE results across trials
            kld_all_trials: Complexity results across trials
            accur_all_trials: Accuracy results across trials
        """
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))

        # VFE plot
        std_val = 1.96 * vfe_all_trials.std(axis=0)
        mean_val = vfe_all_trials.mean(axis=0)
        axes[0].fill_between(po_vec, mean_val + std_val, mean_val - std_val, alpha=0.5)
        axes[0].plot(po_vec, mean_val, label="$\\mathcal{F}$", linewidth=2)
        axes[0].legend(loc="lower right", fontsize=14)
        axes[0].set_ylabel("$\\hat{\\mathcal{F}}$ (Variational Free Energy)", fontsize=14)
        axes[0].grid(True, alpha=0.3)

        # Components plot
        std_kld = 1.96 * kld_all_trials.std(axis=0)
        mean_kld = kld_all_trials.mean(axis=0)
        axes[1].fill_between(po_vec, mean_kld + std_kld, mean_kld - std_kld, alpha=0.5)
        axes[1].plot(po_vec, mean_kld, label="Complexity", linewidth=2)

        std_acc = 1.96 * accur_all_trials.std(axis=0)
        mean_acc = -accur_all_trials.mean(axis=0)
        axes[1].fill_between(po_vec, mean_acc + std_acc, mean_acc - std_acc, alpha=0.5)
        axes[1].plot(po_vec, mean_acc, label="(-ve) Accuracy", linewidth=2)

        axes[1].legend(loc="lower right", fontsize=14)
        axes[1].set_xlabel("Likelihood parameter: $p_{\\mathcal{O}}$", fontsize=14)
        axes[1].set_ylabel("Average nats", fontsize=14)
        axes[1].grid(True, alpha=0.3)

        for ax in axes:
            ax.tick_params(axis="both", which="major", labelsize=12)
            ax.set_xlim(po_vec[0], po_vec[-1])

        plt.tight_layout()

    def calculate_branching_parameter(self, phi_hist: np.ndarray,
                                    transient_cutoff: int = 100) -> float:
        """
        Calculate branching parameter using MR estimator.

        Args:
            phi_hist: History of posterior beliefs
            transient_cutoff: Number of initial time steps to exclude

        Returns:
            Branching parameter estimate
        """
        # Exclude transient behavior
        average_activity = phi_hist[:, transient_cutoff:].mean(axis=0)

        if len(average_activity) < 2:
            return np.nan

        a_next = average_activity[1:]
        a_past = average_activity[:-1]
        a_past_centered = a_past - a_past.mean()

        numerator = (a_next - a_next.mean()).T @ a_past_centered
        denominator = (a_past_centered ** 2).sum()

        if denominator == 0:
            return np.nan

        return numerator / denominator

    def plot_network_visualization(self, spin_state: np.ndarray,
                                 pos: Optional[dict] = None,
                                 node_size: int = 50,
                                 save_path: str = None):
        """
        Visualize the network structure with current spin states as node colors.

        Args:
            spin_state: Current spin configuration
            pos: Node positions (if None, will compute using spring layout)
            node_size: Size of nodes in visualization
            save_path: Path to save the figure (optional)
        """
        if self.graph is None:
            raise ValueError("Network not created. Call create_network() first.")

        plt.figure(figsize=(12, 10))

        if pos is None:
            pos = nx.spring_layout(self.graph, k=1, iterations=50)

        # Color nodes by spin state
        node_colors = ['red' if spin == 1 else 'blue' for spin in spin_state]

        nx.draw(self.graph, pos, node_color=node_colors, node_size=node_size,
                with_labels=False, edge_color='gray', alpha=0.7, width=0.5)

        plt.title("Network Structure with Spin States\n(Red=Down, Blue=Up)", fontsize=14)
        plt.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network visualization saved to {save_path}")

    def plot_phase_space_trajectory(self, phi_hist: np.ndarray,
                                  nodes_to_plot: list = None,
                                  time_window: Tuple[int, int] = None,
                                  save_path: str = None):
        """
        Plot phase space trajectories for selected nodes.

        Args:
            phi_hist: History of posterior beliefs
            nodes_to_plot: List of node indices to plot (if None, plot first 5)
            time_window: Tuple of (start, end) time indices
            save_path: Path to save the figure (optional)
        """
        if nodes_to_plot is None:
            nodes_to_plot = list(range(min(5, self.n_nodes)))

        if time_window is None:
            time_window = (0, phi_hist.shape[1])

        start_t, end_t = time_window

        plt.figure(figsize=(12, 8))

        for i, node in enumerate(nodes_to_plot):
            trajectory = phi_hist[node, start_t:end_t]
            time_points = np.arange(start_t, end_t)

            # Create a color gradient for the trajectory
            colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory)))

            for j in range(len(trajectory) - 1):
                plt.plot([time_points[j], time_points[j+1]],
                        [trajectory[j], trajectory[j+1]],
                        color=colors[j], alpha=0.7, linewidth=2)

            # Mark start and end points
            plt.scatter(time_points[0], trajectory[0],
                       color='green', s=100, marker='o',
                       label=f'Node {node} start' if i == 0 else "")
            plt.scatter(time_points[-1], trajectory[-1],
                       color='red', s=100, marker='s',
                       label=f'End points' if i == 0 else "")

        plt.xlabel("Time", fontsize=14)
        plt.ylabel("Posterior Belief (φ)", fontsize=14)
        plt.title("Phase Space Trajectories", fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Phase space trajectories saved to {save_path}")

    def plot_correlation_matrix(self, spin_hist: np.ndarray,
                              time_window: Tuple[int, int] = None,
                              save_path: str = None):
        """
        Plot correlation matrix between nodes' activities.

        Args:
            spin_hist: History of spin states
            time_window: Time window to compute correlations over
            save_path: Path to save the figure (optional)
        """
        if time_window is None:
            time_window = (0, spin_hist.shape[1])

        start_t, end_t = time_window
        data = spin_hist[:, start_t:end_t]

        # Subsample nodes if too many for visualization
        max_nodes = 100
        if self.n_nodes > max_nodes:
            indices = np.linspace(0, self.n_nodes-1, max_nodes, dtype=int)
            data = data[indices, :]
            title_suffix = f" (subsampled {max_nodes} nodes)"
        else:
            title_suffix = ""

        correlation_matrix = np.corrcoef(data)

        plt.figure(figsize=(10, 8))
        im = plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, label='Correlation Coefficient')
        plt.title(f"Spin-Spin Correlation Matrix{title_suffix}", fontsize=16)
        plt.xlabel("Node Index", fontsize=14)
        plt.ylabel("Node Index", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix saved to {save_path}")

    def plot_avalanche_analysis(self, spin_hist: np.ndarray, threshold: float = 0.1,
                              save_path: str = None):
        """
        Analyze and plot avalanche dynamics (cascades of spin flips).

        Args:
            spin_hist: History of spin states
            threshold: Threshold for detecting significant activity
            save_path: Path to save the figure (optional)
        """
        # Calculate activity (number of nodes that flipped)
        flips = np.abs(np.diff(spin_hist, axis=1))
        avalanche_sizes = flips.sum(axis=0)

        # Find avalanche events
        active_times = avalanche_sizes > threshold * self.n_nodes

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Avalanche size time series
        axes[0, 0].plot(avalanche_sizes)
        axes[0, 0].axhline(y=threshold * self.n_nodes, color='red',
                          linestyle='--', label=f'Threshold ({threshold:.1%})')
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Avalanche Size")
        axes[0, 0].set_title("Avalanche Size Over Time")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Avalanche size distribution
        non_zero_avalanches = avalanche_sizes[avalanche_sizes > 0]
        if len(non_zero_avalanches) > 0:
            axes[0, 1].hist(non_zero_avalanches, bins=50, alpha=0.7, density=True)
            axes[0, 1].set_xlabel("Avalanche Size")
            axes[0, 1].set_ylabel("Probability Density")
            axes[0, 1].set_title("Avalanche Size Distribution")
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)

        # Activity heatmap around avalanches
        if np.any(active_times):
            # Find the largest avalanche
            max_avalanche_time = np.argmax(avalanche_sizes)
            window = 50  # time window around avalanche
            start_idx = max(0, max_avalanche_time - window)
            end_idx = min(spin_hist.shape[1], max_avalanche_time + window)

            axes[1, 0].imshow(spin_hist[:, start_idx:end_idx],
                            aspect='auto', cmap='gray', interpolation='none')
            axes[1, 0].axvline(x=max_avalanche_time - start_idx,
                             color='red', linewidth=2, label='Max Avalanche')
            axes[1, 0].set_xlabel("Time")
            axes[1, 0].set_ylabel("Node")
            axes[1, 0].set_title("Activity Around Largest Avalanche")
            axes[1, 0].legend()

        # Autocorrelation of avalanche sizes
        if len(avalanche_sizes) > 10:
            max_lag = min(100, len(avalanche_sizes) // 4)
            autocorr = np.correlate(avalanche_sizes - avalanche_sizes.mean(),
                                  avalanche_sizes - avalanche_sizes.mean(),
                                  mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize

            lags = np.arange(max_lag)
            axes[1, 1].plot(lags, autocorr[:max_lag])
            axes[1, 1].set_xlabel("Lag")
            axes[1, 1].set_ylabel("Autocorrelation")
            axes[1, 1].set_title("Avalanche Size Autocorrelation")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Avalanche analysis saved to {save_path}")

    def plot_energy_landscape(self, spin_hist: np.ndarray, phi_hist: np.ndarray,
                            time_window: Tuple[int, int] = None,
                            save_path: str = None):
        """
        Plot the energy landscape and system trajectory through it.

        Args:
            spin_hist: History of spin states
            phi_hist: History of posterior beliefs
            time_window: Time window for analysis
            save_path: Path to save the figure (optional)
        """
        if time_window is None:
            time_window = (0, min(500, spin_hist.shape[1]))  # Limit for computation

        start_t, end_t = time_window

        # Calculate system magnetization and energy-like quantities
        magnetization = spin_hist[:, start_t:end_t].mean(axis=0)
        avg_posterior = phi_hist[:, start_t:end_t].mean(axis=0)

        # Calculate "energy" based on neighbor interactions
        energies = []
        for t in range(start_t, end_t):
            spins = spin_hist[:, t]
            # Convert to +1/-1 representation
            ising_spins = 2 * spins - 1
            # Calculate interaction energy
            energy = -0.5 * ising_spins.T @ self.adjacency_matrix @ ising_spins
            energies.append(energy)

        energies = np.array(energies)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Energy vs time
        axes[0, 0].plot(range(len(energies)), energies)
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("System Energy")
        axes[0, 0].set_title("Energy Evolution")
        axes[0, 0].grid(True, alpha=0.3)

        # Magnetization vs time
        axes[0, 1].plot(range(len(magnetization)), magnetization)
        axes[0, 1].set_xlabel("Time")
        axes[0, 1].set_ylabel("Magnetization")
        axes[0, 1].set_title("Magnetization Evolution")
        axes[0, 1].grid(True, alpha=0.3)

        # Phase space: Energy vs Magnetization
        scatter = axes[1, 0].scatter(magnetization, energies,
                                   c=range(len(energies)),
                                   cmap='viridis', alpha=0.7)
        axes[1, 0].set_xlabel("Magnetization")
        axes[1, 0].set_ylabel("Energy")
        axes[1, 0].set_title("Energy-Magnetization Phase Space")
        plt.colorbar(scatter, ax=axes[1, 0], label='Time')

        # Energy distribution
        axes[1, 1].hist(energies, bins=30, alpha=0.7, density=True)
        axes[1, 1].set_xlabel("Energy")
        axes[1, 1].set_ylabel("Probability Density")
        axes[1, 1].set_title("Energy Distribution")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Energy landscape saved to {save_path}")

        energies = np.array(energies)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Energy vs time
        axes[0, 0].plot(range(len(energies)), energies)
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("System Energy")
        axes[0, 0].set_title("Energy Evolution")
        axes[0, 0].grid(True, alpha=0.3)

        # Magnetization vs time
        axes[0, 1].plot(range(len(magnetization)), magnetization)
        axes[0, 1].set_xlabel("Time")
        axes[0, 1].set_ylabel("Magnetization")
        axes[0, 1].set_title("Magnetization Evolution")
        axes[0, 1].grid(True, alpha=0.3)

        # Phase space: Energy vs Magnetization
        scatter = axes[1, 0].scatter(magnetization, energies,
                                   c=range(len(energies)),
                                   cmap='viridis', alpha=0.7)
        axes[1, 0].set_xlabel("Magnetization")
        axes[1, 0].set_ylabel("Energy")
        axes[1, 0].set_title("Energy-Magnetization Phase Space")
        plt.colorbar(scatter, ax=axes[1, 0], label='Time')

        # Energy distribution
        axes[1, 1].hist(energies, bins=30, alpha=0.7, density=True)
        axes[1, 1].set_xlabel("Energy")
        axes[1, 1].set_ylabel("Probability Density")
        axes[1, 1].set_title("Energy Distribution")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

    def plot_synchronization_analysis(self, phi_hist: np.ndarray, save_path: str = None):
        """
        Analyze and visualize synchronization between nodes.

        Args:
            phi_hist: History of posterior beliefs
            save_path: Path to save the figure (optional)
        """
        # Calculate order parameter (synchronization measure)
        complex_order = np.mean(np.exp(1j * 2 * np.pi * phi_hist), axis=0)
        order_parameter = np.abs(complex_order)

        # Calculate pairwise phase differences for a subset of nodes
        n_sample = min(20, self.n_nodes)
        sample_indices = np.linspace(0, self.n_nodes-1, n_sample, dtype=int)
        sampled_phases = 2 * np.pi * phi_hist[sample_indices, :]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Order parameter over time
        axes[0, 0].plot(order_parameter)
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Order Parameter")
        axes[0, 0].set_title("Global Synchronization")
        axes[0, 0].grid(True, alpha=0.3)

        # Phase dynamics for sample nodes
        for i, idx in enumerate(sample_indices[:5]):
            axes[0, 1].plot(sampled_phases[i, :], alpha=0.7,
                          label=f'Node {idx}' if i < 3 else "")
        axes[0, 1].set_xlabel("Time")
        axes[0, 1].set_ylabel("Phase (radians)")
        axes[0, 1].set_title("Phase Dynamics (Sample Nodes)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Phase difference heatmap - Fixed broadcasting issue
        time_indices = np.arange(0, sampled_phases.shape[1], 10)  # Subsample time
        sampled_time_phases = sampled_phases[:, time_indices]  # Shape: (n_sample, n_time_points)

        # Calculate pairwise phase differences
        n_nodes_sample, n_time_sample = sampled_time_phases.shape
        phase_diffs = np.zeros((n_nodes_sample, n_nodes_sample, n_time_sample))

        for t in range(n_time_sample):
            for i in range(n_nodes_sample):
                for j in range(n_nodes_sample):
                    phase_diffs[i, j, t] = np.abs(sampled_time_phases[i, t] - sampled_time_phases[j, t])

        mean_phase_diffs = np.mean(phase_diffs, axis=2)

        im = axes[1, 0].imshow(mean_phase_diffs, cmap='viridis')
        axes[1, 0].set_xlabel("Node Index")
        axes[1, 0].set_ylabel("Node Index")
        axes[1, 0].set_title("Mean Phase Differences")
        plt.colorbar(im, ax=axes[1, 0])

        # Order parameter distribution
        axes[1, 1].hist(order_parameter, bins=30, alpha=0.7, density=True)
        axes[1, 1].set_xlabel("Order Parameter")
        axes[1, 1].set_ylabel("Probability Density")
        axes[1, 1].set_title("Synchronization Distribution")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Synchronization analysis saved to {save_path}")


def main():
    """
    Main function demonstrating the usage of the IsingNetworkSimulator.
    """
    import os

    # Create output directory for saved figures
    output_dir = "ising_simulation_outputs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving figures to: {os.path.abspath(output_dir)}")

    # Initialize simulator
    simulator = IsingNetworkSimulator(n_nodes=100, connection_prob=0.05)  # Smaller for demos

    # Create network
    simulator.create_network('erdos_renyi')

    # Example 1: Single simulation with network visualization
    print("Running single simulation...")
    initial_spins, initial_posteriors = simulator.initialize_system()
    phi_hist, spin_hist, kld_hist, accur_hist = simulator.run_simulation(
        time_steps=500,
        initial_spins=initial_spins,
        initial_posteriors=initial_posteriors,
        po=0.65
    )

    # Network visualization at final state
    network_path = os.path.join(output_dir, "network_visualization.png")
    simulator.plot_network_visualization(spin_hist[:, -1], save_path=network_path)
    plt.show()

    # Phase space trajectories
    phase_path = os.path.join(output_dir, "phase_trajectories.png")
    simulator.plot_phase_space_trajectory(phi_hist, nodes_to_plot=[0, 1, 2, 3, 4],
                                        save_path=phase_path)
    plt.show()

    # Correlation analysis
    corr_path = os.path.join(output_dir, "correlation_matrix.png")
    simulator.plot_correlation_matrix(spin_hist, time_window=(100, 500),
                                    save_path=corr_path)
    plt.show()

    # Avalanche analysis
    avalanche_path = os.path.join(output_dir, "avalanche_analysis.png")
    simulator.plot_avalanche_analysis(spin_hist, save_path=avalanche_path)
    plt.show()

    # Energy landscape
    energy_path = os.path.join(output_dir, "energy_landscape.png")
    simulator.plot_energy_landscape(spin_hist, phi_hist, save_path=energy_path)
    plt.show()

    # Synchronization analysis
    sync_path = os.path.join(output_dir, "synchronization_analysis.png")
    simulator.plot_synchronization_analysis(phi_hist, save_path=sync_path)
    plt.show()

    # Activity heatmap
    activity_path = os.path.join(output_dir, "activity_heatmap.png")
    simulator.plot_activity_heatmap(spin_hist, "Activity Heatmap")
    plt.savefig(activity_path, dpi=300, bbox_inches='tight')
    print(f"Activity heatmap saved to {activity_path}")
    plt.show()

    # Regime comparison
    print("Comparing different regimes...")
    regime_path = os.path.join(output_dir, "regime_comparison.png")
    po_values = [0.50001, 0.601, 0.75]
    simulator.plot_regime_comparison(po_values)
    plt.savefig(regime_path, dpi=300, bbox_inches='tight')
    print(f"Regime comparison saved to {regime_path}")
    plt.show()

    print(f"\n🎨 Cool visualizations complete!")
    print(f"📁 All figures saved to: {os.path.abspath(output_dir)}")
    print("🔬 Check out the avalanche distributions and synchronization patterns!")


if __name__ == "__main__":
    main()
