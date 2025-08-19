"""
Exponential Term Analysis for Dynamic Systems

This module provides analysis tools for computing and visualizing exponential/softmax terms
commonly found in dynamic systems, particularly in variational inference and active inference models.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, List
from dataclasses import dataclass
import seaborn as sns


@dataclass
class AnalysisParameters:
    """Configuration parameters for the exponential term analysis."""
    omega_range: Tuple[float, float] = (0.0, 1.0)
    omega_points: int = 20
    k_range: Tuple[float, float] = (0.0, 5.0)
    k_points: int = 10


class ExponentialTermAnalyzer:
    """
    Analyzer for exponential/softmax terms in dynamic systems.

    This class provides methods to compute, analyze, and visualize exponential terms
    that commonly appear in variational inference and active inference frameworks.
    """

    def __init__(self, params: Optional[AnalysisParameters] = None):
        """
        Initialize the analyzer.

        Args:
            params: Analysis parameters (uses defaults if None)
        """
        self.params = params or AnalysisParameters()
        self.omega_vec = None
        self.k_vec = None
        self.evaluation_grid = None

    def setup_parameter_space(self):
        """Setup the parameter space for omega and k values."""
        self.omega_vec = np.linspace(
            self.params.omega_range[0],
            self.params.omega_range[1],
            self.params.omega_points
        )

        self.k_vec = np.linspace(
            self.params.k_range[0],
            self.params.k_range[1],
            self.params.k_points
        )

    @staticmethod
    def compute_exponential_term(omega: Union[float, np.ndarray],
                               k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the exponential/softmax term for given omega and k values.

        This function computes: 2 * (omega + omega_C * exp(k * (1 - 2*omega))) / (1 + exp(k * (1 - 2*omega)))
        where omega_C = 1 - omega.

        Args:
            omega: Omega parameter value(s) (typically in [0, 1])
            k: Precision parameter value(s) (typically >= 0)

        Returns:
            Computed exponential term value(s)
        """
        omega_c = 1.0 - omega
        exp_term = np.exp(k * (1.0 - 2 * omega))
        numerator = omega + omega_c * exp_term
        denominator = 1.0 + exp_term
        return 2 * (numerator / denominator)

    @staticmethod
    def compute_exponential_term_alternative(omega: Union[float, np.ndarray],
                                           k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Alternative formulation of the exponential term (commented out in original).

        This computes: -2 * (omega * exp(k*omega) + omega_C * exp(k*omega_C)) / (exp(k*omega) + exp(k*omega_C))

        Args:
            omega: Omega parameter value(s)
            k: Precision parameter value(s)

        Returns:
            Computed exponential term value(s)
        """
        omega_c = 1.0 - omega
        exp_k_omega = np.exp(k * omega)
        exp_k_omega_c = np.exp(k * omega_c)
        numerator = omega * exp_k_omega + omega_c * exp_k_omega_c
        denominator = exp_k_omega + exp_k_omega_c
        return -2 * (numerator / denominator)

    def compute_evaluation_grid(self) -> np.ndarray:
        """
        Compute the exponential term over the entire parameter grid.

        Returns:
            2D array with shape (len(omega_vec), len(k_vec)) containing computed values
        """
        if self.omega_vec is None or self.k_vec is None:
            self.setup_parameter_space()

        self.evaluation_grid = np.empty((len(self.omega_vec), len(self.k_vec)))

        for i, omega in enumerate(self.omega_vec):
            for j, k in enumerate(self.k_vec):
                self.evaluation_grid[i, j] = self.compute_exponential_term(omega, k)

        return self.evaluation_grid

    def plot_omega_dependence(self,
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 8),
                            style: str = 'seaborn-v0_8') -> plt.Figure:
        """
        Plot the exponential term as a function of omega for different k values.

        Args:
            save_path: Path to save the figure (optional)
            figsize: Figure size tuple
            style: Matplotlib style to use

        Returns:
            Figure object
        """
        if self.evaluation_grid is None:
            self.compute_evaluation_grid()

        plt.style.use(style) if style in plt.style.available else None

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Create color palette
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.k_vec)))

        for j, k in enumerate(self.k_vec):
            ax.plot(self.omega_vec, self.evaluation_grid[:, j],
                   label=f"k = {k:.3f}",
                   color=colors[j],
                   linewidth=2.5,
                   alpha=0.8)

        ax.set_xlabel("Omega (ω)", fontsize=14)
        ax.set_ylabel("Exponential Term Value", fontsize=14)
        ax.set_title("Exponential Term as Function of Omega\nfor Different Precision Values (k)", fontsize=16)
        ax.legend(loc="best", fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Omega dependence plot saved to {save_path}")

        return fig

    def plot_k_dependence(self,
                         omega_values: Optional[List[float]] = None,
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot the exponential term as a function of k for different omega values.

        Args:
            omega_values: Specific omega values to plot (if None, uses evenly spaced values)
            save_path: Path to save the figure (optional)
            figsize: Figure size tuple

        Returns:
            Figure object
        """
        if self.evaluation_grid is None:
            self.compute_evaluation_grid()

        if omega_values is None:
            # Select evenly spaced omega values
            n_omega_lines = min(8, len(self.omega_vec))
            omega_indices = np.linspace(0, len(self.omega_vec)-1, n_omega_lines, dtype=int)
            omega_values = self.omega_vec[omega_indices]
        else:
            # Find closest indices for specified omega values
            omega_indices = [np.argmin(np.abs(self.omega_vec - omega)) for omega in omega_values]

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        colors = plt.cm.plasma(np.linspace(0, 1, len(omega_values)))

        for i, (omega_idx, omega_val) in enumerate(zip(omega_indices, omega_values)):
            ax.plot(self.k_vec, self.evaluation_grid[omega_idx, :],
                   label=f"ω = {omega_val:.3f}",
                   color=colors[i],
                   linewidth=2.5,
                   alpha=0.8)

        ax.set_xlabel("Precision Parameter (k)", fontsize=14)
        ax.set_ylabel("Exponential Term Value", fontsize=14)
        ax.set_title("Exponential Term as Function of Precision Parameter\nfor Different Omega Values", fontsize=16)
        ax.legend(loc="best", fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"K dependence plot saved to {save_path}")

        return fig

    def plot_heatmap(self,
                    save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 8),
                    cmap: str = 'RdYlBu_r') -> plt.Figure:
        """
        Plot a heatmap of the exponential term over the (omega, k) parameter space.

        Args:
            save_path: Path to save the figure (optional)
            figsize: Figure size tuple
            cmap: Colormap for the heatmap

        Returns:
            Figure object
        """
        if self.evaluation_grid is None:
            self.compute_evaluation_grid()

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Create meshgrid for proper heatmap orientation
        K, Omega = np.meshgrid(self.k_vec, self.omega_vec)

        im = ax.contourf(K, Omega, self.evaluation_grid, levels=50, cmap=cmap)

        # Add contour lines
        contours = ax.contour(K, Omega, self.evaluation_grid, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Exponential Term Value', fontsize=12)

        ax.set_xlabel("Precision Parameter (k)", fontsize=14)
        ax.set_ylabel("Omega (ω)", fontsize=14)
        ax.set_title("Exponential Term Heatmap\nover (k, ω) Parameter Space", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")

        return fig

    def plot_3d_surface(self,
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (12, 9),
                       cmap: str = 'viridis',
                       elevation: float = 30,
                       azimuth: float = 45) -> plt.Figure:
        """
        Plot a 3D surface of the exponential term.

        Args:
            save_path: Path to save the figure (optional)
            figsize: Figure size tuple
            cmap: Colormap for the surface
            elevation: Viewing elevation angle
            azimuth: Viewing azimuth angle

        Returns:
            Figure object
        """
        if self.evaluation_grid is None:
            self.compute_evaluation_grid()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Create meshgrid
        K, Omega = np.meshgrid(self.k_vec, self.omega_vec)

        # Plot surface
        surf = ax.plot_surface(K, Omega, self.evaluation_grid,
                             cmap=cmap, alpha=0.9, linewidth=0, antialiased=True)

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Exponential Term Value')

        ax.set_xlabel("Precision Parameter (k)", fontsize=12)
        ax.set_ylabel("Omega (ω)", fontsize=12)
        ax.set_zlabel("Exponential Term Value", fontsize=12)
        ax.set_title("3D Surface: Exponential Term over (k, ω) Space", fontsize=14)

        # Set viewing angle
        ax.view_init(elev=elevation, azim=azimuth)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D surface plot saved to {save_path}")

        return fig

    def generate_comprehensive_analysis(self,
                                      output_dir: str = "exponential_analysis_output") -> None:
        """
        Generate a comprehensive analysis with all visualizations.

        Args:
            output_dir: Directory to save all plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"Generating comprehensive analysis in {output_dir}/")

        # Compute the evaluation grid
        self.compute_evaluation_grid()

        # Generate all plots
        self.plot_omega_dependence(save_path=f"{output_dir}/omega_dependence.png")
        plt.show()

        self.plot_k_dependence(save_path=f"{output_dir}/k_dependence.png")
        plt.show()

        self.plot_heatmap(save_path=f"{output_dir}/parameter_heatmap.png")
        plt.show()

        self.plot_3d_surface(save_path=f"{output_dir}/3d_surface.png")
        plt.show()

        # Save numerical results
        np.save(f"{output_dir}/omega_vec.npy", self.omega_vec)
        np.save(f"{output_dir}/k_vec.npy", self.k_vec)
        np.save(f"{output_dir}/evaluation_grid.npy", self.evaluation_grid)

        print("✅ Comprehensive analysis complete!")
        print(f"📁 All files saved to: {os.path.abspath(output_dir)}")

    def compute_derivatives(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute numerical derivatives of the exponential term.

        Returns:
            Tuple of (d/domega, d/dk) derivative grids
        """
        if self.evaluation_grid is None:
            self.compute_evaluation_grid()

        # Compute derivatives using central differences
        d_omega = np.gradient(self.evaluation_grid, self.omega_vec, axis=0)
        d_k = np.gradient(self.evaluation_grid, self.k_vec, axis=1)

        return d_omega, d_k

    def find_critical_points(self, tolerance: float = 1e-6) -> List[Tuple[float, float]]:
        """
        Find critical points where both partial derivatives are approximately zero.

        Args:
            tolerance: Tolerance for considering derivatives as zero

        Returns:
            List of (omega, k) tuples for critical points
        """
        d_omega, d_k = self.compute_derivatives()

        # Find points where both derivatives are small
        critical_mask = (np.abs(d_omega) < tolerance) & (np.abs(d_k) < tolerance)
        critical_indices = np.where(critical_mask)

        critical_points = []
        for i, j in zip(critical_indices[0], critical_indices[1]):
            omega_val = self.omega_vec[i]
            k_val = self.k_vec[j]
            critical_points.append((omega_val, k_val))

        return critical_points


def main():
    """
    Main function demonstrating the exponential term analysis.
    """
    print("Starting Exponential Term Analysis...")

    # Create analyzer with custom parameters
    params = AnalysisParameters(
        omega_range=(0.0, 1.0),
        omega_points=50,  # Higher resolution
        k_range=(0.0, 5.0),
        k_points=25      # Higher resolution
    )

    analyzer = ExponentialTermAnalyzer(params)

    # Generate comprehensive analysis
    analyzer.generate_comprehensive_analysis()

    # Example of specific analysis
    print("Computing critical points...")
    critical_points = analyzer.find_critical_points(tolerance=0.01)
    if critical_points:
        print(f"Found {len(critical_points)} critical points:")
        for i, (omega, k) in enumerate(critical_points):
            print(f"  {i+1}. ω = {omega:.4f}, k = {k:.4f}")
    else:
        print("No critical points found with current tolerance.")

    # Example of computing specific values
    print("Example computations:")
    test_omega = 0.5
    test_k = 2.0
    result = analyzer.compute_exponential_term(test_omega, test_k)
    print(f"f(ω={test_omega}, k={test_k}) = {result:.6f}")

    print("Analysis complete!")


if __name__ == "__main__":
    main()
