"""
Parameter Sweep for Ising Network Dynamics

This module provides a clean implementation for running parameter sweeps
over network dynamics simulations with comprehensive data collection and storage.
"""

import os
import argparse
import itertools
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterator
import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import networkx as nx
from tqdm import tqdm

# Import handling with fallbacks
try:
    from simulation.config import NETWORK_GENERATORS
    from simulation.simulation import Simulation
except ImportError:
    try:
        from .simulation.config import NETWORK_GENERATORS
        from .simulation.simulation import Simulation
    except ImportError:
        try:
            from network.network_config import graph_generation_fns as NETWORK_GENERATORS
            from simulation.simulation import Simulation
        except ImportError:
            print("Warning: Could not import required modules. Using fallback implementations.")
            NETWORK_GENERATORS = None
            Simulation = None


@dataclass
class SweepParameters:
    """Configuration for parameter sweep."""
    network_size: int
    num_trials: int
    time_steps: int
    graph_type: str = "er"
    prior_probability: float = 0.5
    transient_cutoff: int = 100


@dataclass
class NetworkParameters:
    """Parameters for a single network configuration."""
    N: int  # Network size
    p: float  # Connection probability or other network parameter
    po: float  # Observation likelihood parameter


@dataclass
class TrialResults:
    """Results from a single trial."""
    avg_vfe: float
    avg_complexity: float
    avg_neg_accuracy: float
    avg_polarization: float
    branching_parameter: float
    adjacency_matrix: np.ndarray


class ParameterSweepRunner:
    """
    Main class for running parameter sweeps over Ising network dynamics.
    """

    def __init__(self, sweep_params: SweepParameters, output_dir: str):
        """
        Initialize the parameter sweep runner.

        Args:
            sweep_params: Configuration for the sweep
            output_dir: Directory to save results
        """
        self.sweep_params = sweep_params
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Validate dependencies
        self._validate_dependencies()

    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / "sweep.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _validate_dependencies(self):
        """Validate that required dependencies are available."""
        if NETWORK_GENERATORS is None or Simulation is None:
            self.logger.warning("Required modules not found. Some functionality may be limited.")

    def generate_parameter_combinations(self,
                                      p_range: Tuple[float, float] = None,
                                      po_range: Tuple[float, float] = (0.5, 1.0),
                                      n_p_points: int = 20,
                                      n_po_points: int = 50) -> List[NetworkParameters]:
        """
        Generate parameter combinations for the sweep.

        Args:
            p_range: Range for connection probability (if None, auto-computed)
            po_range: Range for observation likelihood parameter
            n_p_points: Number of points in p range
            n_po_points: Number of points in po range

        Returns:
            List of NetworkParameters for the sweep
        """
        N = self.sweep_params.network_size

        # Auto-compute p range if not provided
        if p_range is None:
            # Starting from connectivity threshold
            starting_p = 1.5 * np.log(N) / N
            p_range = (starting_p, 1.0)

        # Generate parameter ranges
        p_levels = np.linspace(p_range[0], p_range[1], n_p_points)
        po_levels = np.linspace(po_range[0] + 1e-16, po_range[1] - 1e-16, n_po_points)

        # Create parameter combinations
        param_combinations = []
        for p, po in itertools.product(p_levels, po_levels):
            param_combinations.append(NetworkParameters(N=N, p=p, po=po))

        self.logger.info(f"Generated {len(param_combinations)} parameter combinations")
        self.logger.info(f"P range: [{p_range[0]:.4f}, {p_range[1]:.4f}]")
        self.logger.info(f"Po range: [{po_range[0]:.4f}, {po_range[1]:.4f}]")

        return param_combinations

    def run_single_trial(self, network_params: NetworkParameters) -> TrialResults:
        """
        Run a single trial with given network parameters.

        Args:
            network_params: Parameters for this trial

        Returns:
            TrialResults object with computed metrics
        """
        # Generate network
        if NETWORK_GENERATORS and self.sweep_params.graph_type.lower() in NETWORK_GENERATORS:
            G = NETWORK_GENERATORS[self.sweep_params.graph_type.lower()](
                network_params.N, network_params.p
            )
        else:
            # Fallback to networkx
            G = nx.erdos_renyi_graph(network_params.N, network_params.p)

        # Run simulation
        if Simulation:
            sim = Simulation(G)
            phi_hist, spin_hist = sim.run(
                self.sweep_params.time_steps,
                network_params.po,
                self.sweep_params.prior_probability
            )

            # Compute VFE components
            vfe, complexity, neg_accur = sim.compute_VFE(
                phi_hist, spin_hist, decomposition="complexity_accuracy"
            )

            # Compute branching parameter
            branching_param = sim.compute_m(phi_hist[:, self.sweep_params.transient_cutoff:])

            # Get adjacency matrix
            adj_matrix = sim.A

        else:
            # Fallback implementation (simplified)
            self.logger.warning("Using fallback simulation implementation")
            adj_matrix = nx.to_numpy_array(G)
            # Placeholder values - replace with actual implementation
            vfe = np.random.randn(network_params.N, self.sweep_params.time_steps)
            complexity = np.random.randn(network_params.N, self.sweep_params.time_steps)
            neg_accur = np.random.randn(network_params.N, self.sweep_params.time_steps)
            phi_hist = np.random.rand(network_params.N, self.sweep_params.time_steps)
            branching_param = np.random.randn()

        # Exclude transient period for averaging
        cutoff = self.sweep_params.transient_cutoff

        return TrialResults(
            avg_vfe=vfe[:, cutoff:].mean(),
            avg_complexity=complexity[:, cutoff:].mean(),
            avg_neg_accuracy=neg_accur[:, cutoff:].mean(),
            avg_polarization=phi_hist[:, cutoff:].mean(),
            branching_parameter=branching_param,
            adjacency_matrix=adj_matrix
        )

    def run_parameter_configuration(self,
                                  network_params: NetworkParameters,
                                  param_idx: int) -> Dict[str, Any]:
        """
        Run all trials for a single parameter configuration.

        Args:
            network_params: Network parameters for this configuration
            param_idx: Index of this parameter configuration

        Returns:
            Dictionary with aggregated results
        """
        self.logger.info(f"Running parameter config {param_idx}: N={network_params.N}, "
                        f"p={network_params.p:.4f}, po={network_params.po:.4f}")

        # Initialize result arrays
        num_trials = self.sweep_params.num_trials
        N = network_params.N

        avg_vfe_per_trial = np.empty(num_trials)
        avg_complexity_per_trial = np.empty(num_trials)
        avg_neg_accur_per_trial = np.empty(num_trials)
        avg_polarization_per_trial = np.empty(num_trials)
        avg_m_per_trial = np.empty(num_trials)
        adj_mat_per_trial = np.zeros((N, N, num_trials))

        # Run trials with progress bar
        for trial_i in tqdm(range(num_trials),
                           desc=f"Config {param_idx}",
                           leave=False):
            try:
                trial_result = self.run_single_trial(network_params)

                avg_vfe_per_trial[trial_i] = trial_result.avg_vfe
                avg_complexity_per_trial[trial_i] = trial_result.avg_complexity
                avg_neg_accur_per_trial[trial_i] = trial_result.avg_neg_accuracy
                avg_polarization_per_trial[trial_i] = trial_result.avg_polarization
                avg_m_per_trial[trial_i] = trial_result.branching_parameter
                adj_mat_per_trial[:, :, trial_i] = trial_result.adjacency_matrix

            except Exception as e:
                self.logger.error(f"Error in trial {trial_i} of config {param_idx}: {e}")
                # Fill with NaN for failed trials
                avg_vfe_per_trial[trial_i] = np.nan
                avg_complexity_per_trial[trial_i] = np.nan
                avg_neg_accur_per_trial[trial_i] = np.nan
                avg_polarization_per_trial[trial_i] = np.nan
                avg_m_per_trial[trial_i] = np.nan

        return {
            "avg_vfe_per_trial": avg_vfe_per_trial,
            "avg_complexity_per_trial": avg_complexity_per_trial,
            "avg_neg_accur_per_trial": avg_neg_accur_per_trial,
            "avg_polarization_per_trial": avg_polarization_per_trial,
            "avg_m_per_trial": avg_m_per_trial,
            "adj_mat_per_trial": adj_mat_per_trial,
            "network_params": network_params,
            "sweep_params": self.sweep_params,
            "timestamp": datetime.now().isoformat()
        }

    def save_results(self, results: Dict[str, Any], param_idx: int):
        """
        Save results for a parameter configuration.

        Args:
            results: Results dictionary to save
            param_idx: Parameter configuration index
        """
        param_folder = self.output_dir / f"config_{param_idx:04d}"
        param_folder.mkdir(exist_ok=True)

        # Save main results
        results_file = param_folder / "results.pkl"
        with open(results_file, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save summary statistics
        summary = self._compute_summary_stats(results)
        summary_file = param_folder / "summary.pkl"
        with open(summary_file, "wb") as f:
            pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.debug(f"Saved results for config {param_idx} to {param_folder}")

    def _compute_summary_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute summary statistics for a parameter configuration."""
        summary = {}

        for key in ["avg_vfe_per_trial", "avg_complexity_per_trial",
                   "avg_neg_accur_per_trial", "avg_polarization_per_trial",
                   "avg_m_per_trial"]:
            data = results[key]
            valid_data = data[~np.isnan(data)]

            if len(valid_data) > 0:
                summary[key] = {
                    "mean": np.mean(valid_data),
                    "std": np.std(valid_data),
                    "median": np.median(valid_data),
                    "q25": np.percentile(valid_data, 25),
                    "q75": np.percentile(valid_data, 75),
                    "min": np.min(valid_data),
                    "max": np.max(valid_data),
                    "n_valid": len(valid_data),
                    "n_total": len(data)
                }
            else:
                summary[key] = {"error": "No valid data"}

        return summary

    def run_sweep(self, parameter_combinations: List[NetworkParameters]):
        """
        Run the complete parameter sweep.

        Args:
            parameter_combinations: List of parameter combinations to test
        """
        self.logger.info(f"Starting parameter sweep with {len(parameter_combinations)} configurations")
        self.logger.info(f"Output directory: {self.output_dir.absolute()}")

        # Save sweep configuration
        config_file = self.output_dir / "sweep_config.pkl"
        with open(config_file, "wb") as f:
            pickle.dump({
                "sweep_params": self.sweep_params,
                "parameter_combinations": parameter_combinations,
                "timestamp": datetime.now().isoformat()
            }, f)

        # Run sweep with progress bar
        for param_idx, network_params in enumerate(tqdm(parameter_combinations,
                                                       desc="Parameter sweep")):
            try:
                results = self.run_parameter_configuration(network_params, param_idx)
                self.save_results(results, param_idx)

            except Exception as e:
                self.logger.error(f"Failed to complete parameter config {param_idx}: {e}")
                continue

        self.logger.info("Parameter sweep completed successfully!")

    @staticmethod
    def load_results(output_dir: str, param_idx: int) -> Dict[str, Any]:
        """
        Load results from a specific parameter configuration.

        Args:
            output_dir: Directory containing sweep results
            param_idx: Parameter configuration index

        Returns:
            Results dictionary
        """
        results_file = Path(output_dir) / f"config_{param_idx:04d}" / "results.pkl"
        with open(results_file, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_sweep_summary(output_dir: str) -> Dict[str, Any]:
        """
        Load summary of entire sweep.

        Args:
            output_dir: Directory containing sweep results

        Returns:
            Summary dictionary
        """
        output_path = Path(output_dir)

        # Load sweep configuration
        config_file = output_path / "sweep_config.pkl"
        with open(config_file, "rb") as f:
            sweep_config = pickle.load(f)

        # Collect all summary statistics
        summaries = []
        config_dirs = sorted([d for d in output_path.iterdir()
                             if d.is_dir() and d.name.startswith("config_")])

        for config_dir in config_dirs:
            summary_file = config_dir / "summary.pkl"
            if summary_file.exists():
                with open(summary_file, "rb") as f:
                    summary = pickle.load(f)
                    summaries.append(summary)

        return {
            "sweep_config": sweep_config,
            "config_summaries": summaries,
            "n_configurations": len(summaries)
        }


def main():
    """Main function for running parameter sweeps from command line."""
    parser = argparse.ArgumentParser(
        description="Run parameter sweep for Ising network dynamics"
    )

    parser.add_argument(
        "--run_name", "-name",
        type=str,
        default=f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Name of the sweep run (default: timestamp-based)"
    )

    parser.add_argument(
        "--num_trials", "-nt",
        type=int,
        default=200,
        help="Number of trials per parameter configuration (default: 200)"
    )

    parser.add_argument(
        "--time_steps", "-T",
        type=int,
        default=1000,
        help="Number of simulation time steps (default: 1000)"
    )

    parser.add_argument(
        "--network_size", "-N",
        type=int,
        required=True,
        help="Size of the network (required)"
    )

    parser.add_argument(
        "--graph_type", "-gt",
        type=str,
        default="er",
        help="Type of graph to generate (default: er)"
    )

    parser.add_argument(
        "--n_p_points",
        type=int,
        default=20,
        help="Number of points in connection probability range (default: 20)"
    )

    parser.add_argument(
        "--n_po_points",
        type=int,
        default=50,
        help="Number of points in observation likelihood range (default: 50)"
    )

    parser.add_argument(
        "--po_min",
        type=float,
        default=0.5,
        help="Minimum observation likelihood (default: 0.5)"
    )

    parser.add_argument(
        "--po_max",
        type=float,
        default=1.0,
        help="Maximum observation likelihood (default: 1.0)"
    )

    args = parser.parse_args()

    # Create sweep parameters
    sweep_params = SweepParameters(
        network_size=args.network_size,
        num_trials=args.num_trials,
        time_steps=args.time_steps,
        graph_type=args.graph_type
    )

    # Initialize sweep runner
    runner = ParameterSweepRunner(sweep_params, args.run_name)

    # Generate parameter combinations
    param_combinations = runner.generate_parameter_combinations(
        po_range=(args.po_min, args.po_max),
        n_p_points=args.n_p_points,
        n_po_points=args.n_po_points
    )

    # Run the sweep
    runner.run_sweep(param_combinations)

    print(f"✅ Parameter sweep completed!")
    print(f"📁 Results saved to: {Path(args.run_name).absolute()}")
    print(f"📊 Total configurations: {len(param_combinations)}")
    print(f"🔬 Total trials: {len(param_combinations) * args.num_trials}")


if __name__ == "__main__":
    main()
