"""
Configuration and shared utilities for Ising model simulations.

This module provides common configuration options, network generation functions,
and utility classes used across different simulation types.
"""

import numpy as np
import networkx as nx
from typing import Dict, Callable, Tuple, Any
from pathlib import Path


class SimulationConfig:
    """Configuration class for Ising model simulations."""

    # Default simulation parameters
    DEFAULT_N_NODES = 500
    DEFAULT_CONNECTION_PROB = 0.01
    DEFAULT_TIME_STEPS = 1000
    DEFAULT_PS_VALUE = 0.5
    DEFAULT_PO_RANGE = (0.5, 0.999)

    # Network generation parameters
    DEFAULT_WS_K = 4
    DEFAULT_WS_P = 0.0
    DEFAULT_CLIQUES_NUM = 16
    DEFAULT_CLIQUE_SIZE = 32

    # Parameter ranges
    DEFAULT_K_RANGE = (0.9, 1.1)
    DEFAULT_OMEGA_RANGE = (0.5, 0.7)
    DEFAULT_INIT_SCALE = 0.7
    DEFAULT_LEARNING_RATE = 0.1

    # Visualization settings
    DEFAULT_DPI = 300
    DEFAULT_FIGSIZE = (12, 8)
    DEFAULT_SUBSAMPLE_NODES = 10  # Show every 10th node in heatmaps

    @classmethod
    def get_output_directory(cls, base_name: str = "tmp_out") -> str:
        """Get or create output directory."""
        output_dir = Path(__file__).parent.parent / base_name
        output_dir.mkdir(exist_ok=True)
        return str(output_dir)


class NetworkGenerator:
    """Factory class for generating different types of networks."""

    @staticmethod
    def erdos_renyi(n_nodes: int, connection_prob: float, **kwargs) -> nx.Graph:
        """Generate Erdős-Rényi random graph."""
        return nx.fast_gnp_random_graph(n_nodes, connection_prob)

    @staticmethod
    def watts_strogatz(n_nodes: int, k: int = None, p: float = None, **kwargs) -> nx.Graph:
        """Generate Watts-Strogatz small-world graph."""
        k = k or SimulationConfig.DEFAULT_WS_K
        p = p or SimulationConfig.DEFAULT_WS_P
        return nx.watts_strogatz_graph(n_nodes, k, p)

    @staticmethod
    def ring_of_cliques(num_cliques: int = None, clique_size: int = None, **kwargs) -> nx.Graph:
        """Generate ring of cliques graph."""
        num_cliques = num_cliques or SimulationConfig.DEFAULT_CLIQUES_NUM
        clique_size = clique_size or SimulationConfig.DEFAULT_CLIQUE_SIZE
        return nx.ring_of_cliques(num_cliques, clique_size)

    @staticmethod
    def grid_2d(n_nodes: int, periodic: bool = True, **kwargs) -> nx.Graph:
        """Generate 2D grid graph."""
        d1 = int(np.sqrt(n_nodes))
        d2 = int(n_nodes / d1)
        return nx.grid_2d_graph(d1, d2, periodic=periodic)

    @staticmethod
    def barabasi_albert(n_nodes: int, m: int = 3, **kwargs) -> nx.Graph:
        """Generate Barabási-Albert preferential attachment graph."""
        return nx.barabasi_albert_graph(n_nodes, m)


# Network generation function registry
NETWORK_GENERATORS: Dict[str, Callable] = {
    'erdos_renyi': NetworkGenerator.erdos_renyi,
    'er': NetworkGenerator.erdos_renyi,  # Alias
    'watts_strogatz': NetworkGenerator.watts_strogatz,
    'ws': NetworkGenerator.watts_strogatz,  # Alias
    'ring_of_cliques': NetworkGenerator.ring_of_cliques,
    'roc': NetworkGenerator.ring_of_cliques,  # Alias
    'grid_2d': NetworkGenerator.grid_2d,
    'grid': NetworkGenerator.grid_2d,  # Alias
    'barabasi_albert': NetworkGenerator.barabasi_albert,
    'ba': NetworkGenerator.barabasi_albert,  # Alias
}


class ParameterSampler:
    """Utility class for sampling simulation parameters."""

    @staticmethod
    def uniform_k_matrix(n_nodes: int, k_range: Tuple[float, float] = None) -> np.ndarray:
        """Sample k_matrix with uniform distribution."""
        k_range = k_range or SimulationConfig.DEFAULT_K_RANGE
        return np.random.uniform(*k_range, size=(n_nodes, n_nodes))

    @staticmethod
    def uniform_omega_matrix(n_nodes: int, omega_range: Tuple[float, float] = None) -> np.ndarray:
        """Sample omega_matrix with uniform distribution."""
        omega_range = omega_range or SimulationConfig.DEFAULT_OMEGA_RANGE
        return np.random.uniform(*omega_range, size=(n_nodes, n_nodes))

    @staticmethod
    def constant_ps_vec(n_nodes: int, ps_value: float = None) -> np.ndarray:
        """Create constant ps vector."""
        ps_value = ps_value or SimulationConfig.DEFAULT_PS_VALUE
        return ps_value * np.ones(n_nodes)

    @staticmethod
    def random_ps_vec(n_nodes: int) -> np.ndarray:
        """Create random ps vector."""
        return np.random.rand(n_nodes)

    @staticmethod
    def linspace_po_values(po_range: Tuple[float, float] = None, n_points: int = 50) -> np.ndarray:
        """Generate linearly spaced po values."""
        po_range = po_range or SimulationConfig.DEFAULT_PO_RANGE
        return np.linspace(po_range[0], po_range[1], n_points)


class VisualizationHelper:
    """Helper class for common visualization operations."""

    @staticmethod
    def setup_figure(nrows: int = 1, ncols: int = 1,
                    figsize: Tuple[float, float] = None,
                    dpi: int = None) -> Tuple[Any, Any]:
        """Setup matplotlib figure with consistent styling."""
        import matplotlib.pyplot as plt

        figsize = figsize or SimulationConfig.DEFAULT_FIGSIZE
        dpi = dpi or SimulationConfig.DEFAULT_DPI

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)

        # Ensure axes is always an array for consistent handling
        if nrows == 1 and ncols == 1:
            axes = [axes]
        elif nrows == 1 or ncols == 1:
            axes = axes.flatten()

        return fig, axes

    @staticmethod
    def save_figure(fig: Any, output_path: str, dpi: int = None) -> None:
        """Save figure with consistent settings."""
        dpi = dpi or SimulationConfig.DEFAULT_DPI
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {output_path}")

    @staticmethod
    def close_figure(fig: Any) -> None:
        """Close figure to free memory."""
        import matplotlib.pyplot as plt
        plt.close(fig)

    @staticmethod
    def apply_common_styling(ax: Any, title: str = None,
                           xlabel: str = None, ylabel: str = None) -> None:
        """Apply common styling to axes."""
        if title:
            ax.set_title(title, fontsize=14)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=10)


def create_network(network_type: str, n_nodes: int = None, **kwargs) -> nx.Graph:
    """
    Convenience function to create networks using the registered generators.

    Args:
        network_type: Type of network to create
        n_nodes: Number of nodes (if applicable)
        **kwargs: Additional parameters for network creation

    Returns:
        NetworkX graph object

    Raises:
        ValueError: If network_type is not recognized
    """
    if network_type not in NETWORK_GENERATORS:
        available_types = list(NETWORK_GENERATORS.keys())
        raise ValueError(f"Unknown network type '{network_type}'. "
                        f"Available types: {available_types}")

    generator = NETWORK_GENERATORS[network_type]

    # Handle special cases that don't use n_nodes directly
    if network_type in ['ring_of_cliques', 'roc']:
        return generator(**kwargs)
    else:
        n_nodes = n_nodes or SimulationConfig.DEFAULT_N_NODES
        return generator(n_nodes, **kwargs)


def get_default_parameters(network_type: str = 'erdos_renyi',
                          n_nodes: int = None) -> Dict[str, Any]:
    """
    Get default parameters for a given network type.

    Args:
        network_type: Type of network
        n_nodes: Number of nodes

    Returns:
        Dictionary of default parameters
    """
    n_nodes = n_nodes or SimulationConfig.DEFAULT_N_NODES

    base_params = {
        'n_nodes': n_nodes,
        'time_steps': SimulationConfig.DEFAULT_TIME_STEPS,
        'ps_value': SimulationConfig.DEFAULT_PS_VALUE,
        'k_range': SimulationConfig.DEFAULT_K_RANGE,
        'omega_range': SimulationConfig.DEFAULT_OMEGA_RANGE,
        'init_scale': SimulationConfig.DEFAULT_INIT_SCALE,
        'learning_rate': SimulationConfig.DEFAULT_LEARNING_RATE,
    }

    # Network-specific parameters
    if network_type in ['erdos_renyi', 'er']:
        base_params['connection_prob'] = SimulationConfig.DEFAULT_CONNECTION_PROB
    elif network_type in ['watts_strogatz', 'ws']:
        base_params['k'] = SimulationConfig.DEFAULT_WS_K
        base_params['p'] = SimulationConfig.DEFAULT_WS_P
    elif network_type in ['ring_of_cliques', 'roc']:
        base_params['num_cliques'] = SimulationConfig.DEFAULT_CLIQUES_NUM
        base_params['clique_size'] = SimulationConfig.DEFAULT_CLIQUE_SIZE
    elif network_type in ['grid_2d', 'grid']:
        base_params['periodic'] = True
    elif network_type in ['barabasi_albert', 'ba']:
        base_params['m'] = 3

    return base_params
