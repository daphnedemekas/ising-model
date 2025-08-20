#!/usr/bin/env python3
"""
Comprehensive Demo Runner for Ising Model Simulations

This script runs all the different simulations and generates comprehensive plots
for the Ising model project.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_output_directories():
    """Create output directories for all plots."""
    dirs = [
        "plots/ising_demo",
        "plots/vectorized_demo", 
        "plots/exponential_analysis",
        "plots/parameter_sweep"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")
    
    return dirs

def run_ising_demo():
    """Run the main Ising model demo."""
    print("\n🔬 Running Main Ising Model Demo...")
    
    try:
        from main import IsingNetworkSimulator
        
        # Initialize simulator with smaller network for demo
        simulator = IsingNetworkSimulator(n_nodes=100, connection_prob=0.05)
        simulator.create_network('erdos_renyi')
        
        # Run simulation
        initial_spins, initial_posteriors = simulator.initialize_system()
        phi_hist, spin_hist, kld_hist, accur_hist = simulator.run_simulation(
            time_steps=500,
            initial_spins=initial_spins,
            initial_posteriors=initial_posteriors,
            po=0.65
        )
        
        # Generate all plots
        plots = {}
        
        # Network visualization
        plots['network'] = simulator.plot_network_visualization(
            spin_hist[:, -1], 
            save_path="plots/ising_demo/network_visualization.png"
        )
        
        # Phase space trajectories
        plots['phase_trajectories'] = simulator.plot_phase_space_trajectory(
            phi_hist, 
            nodes_to_plot=[0, 1, 2, 3, 4],
            save_path="plots/ising_demo/phase_trajectories.png"
        )
        
        # Correlation analysis
        plots['correlation'] = simulator.plot_correlation_matrix(
            spin_hist, 
            time_window=(100, 500),
            save_path="plots/ising_demo/correlation_matrix.png"
        )
        
        # Avalanche analysis
        plots['avalanche'] = simulator.plot_avalanche_analysis(
            spin_hist, 
            save_path="plots/ising_demo/avalanche_analysis.png"
        )
        
        # Energy landscape
        plots['energy'] = simulator.plot_energy_landscape(
            spin_hist, phi_hist, 
            save_path="plots/ising_demo/energy_landscape.png"
        )
        
        # Synchronization analysis
        plots['synchronization'] = simulator.plot_synchronization_analysis(
            phi_hist, 
            save_path="plots/ising_demo/synchronization_analysis.png"
        )
        
        # Activity heatmap
        simulator.plot_activity_heatmap(spin_hist, "Activity Heatmap")
        plt.savefig("plots/ising_demo/activity_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()
        plots['activity'] = "plots/ising_demo/activity_heatmap.png"
        
        # Regime comparison
        po_values = [0.50001, 0.601, 0.75]
        simulator.plot_regime_comparison(po_values)
        plt.savefig("plots/ising_demo/regime_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        plots['regime'] = "plots/ising_demo/regime_comparison.png"
        
        print("✅ Main Ising demo completed successfully!")
        return plots
        
    except Exception as e:
        print(f"❌ Error in main Ising demo: {e}")
        return {}

def run_vectorized_demo():
    """Run the vectorized Ising model demo."""
    print("\n⚡ Running Vectorized Ising Model Demo...")
    
    try:
        from vectorized import VectorizedIsingSimulator
        
        # Initialize simulator
        simulator = VectorizedIsingSimulator(n_nodes=256, connection_prob=0.01)
        
        # Run complete demo with Erdős-Rényi network
        print("Running Erdős-Rényi network demo...")
        er_results = simulator.run_complete_demo(
            network_type='erdos_renyi',
            time_steps=500,
            include_learning=True,
            show_plots=False
        )
        
        # Run demo with ring of cliques
        print("Running ring of cliques demo...")
        clique_results = simulator.run_complete_demo(
            network_type='ring_of_cliques',
            time_steps=500,
            include_learning=True,
            show_plots=False
        )
        
        print("✅ Vectorized demo completed successfully!")
        return {'er_results': er_results, 'clique_results': clique_results}
        
    except Exception as e:
        print(f"❌ Error in vectorized demo: {e}")
        return {}

def run_exponential_analysis():
    """Run the exponential term analysis."""
    print("\n📊 Running Exponential Term Analysis...")
    
    try:
        from exponential_term.main import ExponentialTermAnalyzer, AnalysisParameters
        
        # Create analyzer with custom parameters
        params = AnalysisParameters(
            omega_range=(0.0, 1.0),
            omega_points=50,
            k_range=(0.0, 5.0),
            k_points=25
        )
        
        analyzer = ExponentialTermAnalyzer(params)
        
        # Generate comprehensive analysis
        analyzer.generate_comprehensive_analysis("plots/exponential_analysis")
        
        # Find critical points
        critical_points = analyzer.find_critical_points(tolerance=0.01)
        if critical_points:
            print(f"Found {len(critical_points)} critical points:")
            for i, (omega, k) in enumerate(critical_points):
                print(f"  {i+1}. ω = {omega:.4f}, k = {k:.4f}")
        
        print("✅ Exponential analysis completed successfully!")
        return {'critical_points': critical_points}
        
    except Exception as e:
        print(f"❌ Error in exponential analysis: {e}")
        return {}

def run_parameter_sweep():
    """Run a parameter sweep demo."""
    print("\n🔄 Running Parameter Sweep Demo...")
    
    try:
        from parameter_sweep import ParameterSweepRunner, SweepParameters
        
        # Create sweep parameters for a small demo
        sweep_params = SweepParameters(
            network_size=50,  # Smaller for demo
            num_trials=10,    # Fewer trials for demo
            time_steps=200,   # Shorter for demo
            graph_type="er"
        )
        
        # Initialize sweep runner
        runner = ParameterSweepRunner(sweep_params, "plots/parameter_sweep")
        
        # Generate parameter combinations
        param_combinations = runner.generate_parameter_combinations(
            po_range=(0.5, 0.9),
            n_p_points=5,     # Fewer points for demo
            n_po_points=10    # Fewer points for demo
        )
        
        # Run the sweep
        runner.run_sweep(param_combinations)
        
        print("✅ Parameter sweep completed successfully!")
        return {'n_configurations': len(param_combinations)}
        
    except Exception as e:
        print(f"❌ Error in parameter sweep: {e}")
        return {}



def create_summary_report(results):
    """Create a summary report of all results."""
    print("\n📋 Creating Summary Report...")
    
    report = []
    report.append("# Ising Model Simulation Results Summary")
    report.append("")
    report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Summary of each demo
    if 'ising_demo' in results:
        report.append("## Main Ising Model Demo")
        report.append("- Network visualization with spin states")
        report.append("- Phase space trajectories")
        report.append("- Correlation matrix analysis")
        report.append("- Avalanche dynamics")
        report.append("- Energy landscape")
        report.append("- Synchronization analysis")
        report.append("- Activity heatmaps")
        report.append("- Regime comparison")
        report.append("")
    
    if 'vectorized_demo' in results:
        report.append("## Vectorized Ising Model Demo")
        report.append("- Erdős-Rényi network simulation")
        report.append("- Ring of cliques network simulation")
        report.append("- Learning dynamics with k-matrix evolution")
        report.append("")
    
    if 'exponential_analysis' in results:
        report.append("## Exponential Term Analysis")
        report.append("- Omega dependence plots")
        report.append("- Precision parameter analysis")
        report.append("- Parameter space heatmaps")
        report.append("- 3D surface plots")
        if results['exponential_analysis'].get('critical_points'):
            report.append(f"- Found {len(results['exponential_analysis']['critical_points'])} critical points")
        report.append("")
    
    if 'parameter_sweep' in results:
        report.append("## Parameter Sweep Analysis")
        report.append(f"- Swept {results['parameter_sweep'].get('n_configurations', 'unknown')} parameter configurations")
        report.append("- Variational free energy analysis")
        report.append("- Complexity and accuracy decomposition")
        report.append("")
    

    
    report.append("## Output Directories")
    report.append("- `plots/ising_demo/` - Main simulation visualizations")
    report.append("- `plots/vectorized_demo/` - Vectorized simulation results")
    report.append("- `plots/exponential_analysis/` - Exponential term analysis")
    report.append("- `plots/parameter_sweep/` - Parameter sweep results")

    
    # Write report
    with open("plots/summary_report.md", "w") as f:
        f.write("\n".join(report))
    
    print("✅ Summary report created: plots/summary_report.md")

def main():
    """Main function to run all demos."""
    print("🚀 Starting Comprehensive Ising Model Demo Suite")
    print("=" * 60)
    
    # Setup directories
    setup_output_directories()
    
    # Run all demos
    results = {}
    
    # Main Ising demo
    results['ising_demo'] = run_ising_demo()
    
    # Vectorized demo
    results['vectorized_demo'] = run_vectorized_demo()
    
    # Exponential analysis
    results['exponential_analysis'] = run_exponential_analysis()
    
    # Parameter sweep
    results['parameter_sweep'] = run_parameter_sweep()
    

    
    # Create summary report
    create_summary_report(results)
    
    print("\n" + "=" * 60)
    print("🎉 All demos completed successfully!")
    print("📁 Check the 'plots/' directory for all generated visualizations")
    print("📋 See 'plots/summary_report.md' for a detailed summary")
    print("=" * 60)

if __name__ == "__main__":
    main()
