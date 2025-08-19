"""
Ising Model Network Dynamics Package

This package provides tools for simulating Ising-like dynamics on networks
with variational free energy calculations, learning capabilities, and
comprehensive visualization tools.
"""

# Import core simulation classes
try:
    from .simulation import Simulation, SimulationVectorized
except ImportError:
    # Fallback for direct execution
    from simulation import Simulation, SimulationVectorized

# Import configuration and utilities
try:
    from .config import (
        SimulationConfig,
        NetworkGenerator,
        ParameterSampler,
        NETWORK_GENERATORS,
        create_network,
        get_default_parameters
    )
except ImportError:
    from config import (
        SimulationConfig,
        NetworkGenerator,
        ParameterSampler,
        NETWORK_GENERATORS,
        create_network,
        get_default_parameters
    )

# Import visualization tools
try:
    from .visualization import IsingVisualization, quick_plot_activity, quick_plot_comparison
except ImportError:
    from visualization import IsingVisualization, quick_plot_activity, quick_plot_comparison

# Import demo classes
try:
    from .demo_sim import IsingNetworkSimulator
    from .demo_vectorized import VectorizedIsingSimulator
except ImportError:
    try:
        from demo_sim import IsingNetworkSimulator
        from demo_vectorized import VectorizedIsingSimulator
    except ImportError:
        # Demo classes might not be available in all contexts
        IsingNetworkSimulator = None
        VectorizedIsingSimulator = None

# Import math utilities
try:
    from .math_utils import log_stable, compute_exp_normalizing
except ImportError:
    from math_utils import log_stable, compute_exp_normalizing

__version__ = "1.0.0"
__author__ = "Network Active Inference Team"

# Define what gets imported with "from ising_model import *"
__all__ = [
    # Core simulation classes
    'Simulation',
    'SimulationVectorized',

    # Configuration and utilities
    'SimulationConfig',
    'NetworkGenerator',
    'ParameterSampler',
    'NETWORK_GENERATORS',
    'create_network',
    'get_default_parameters',

    # Visualization
    'IsingVisualization',
    'quick_plot_activity',
    'quick_plot_comparison',

    # Demo classes (if available)
    'IsingNetworkSimulator',
    'VectorizedIsingSimulator',

    # Math utilities
    'log_stable',
    'compute_exp_normalizing',
]

# Remove None values from __all__ (for optional imports)
__all__ = [item for item in __all__ if globals().get(item) is not None]
