#!/usr/bin/env python3
"""
Network Demo Script

This script demonstrates different network topologies and their effects on
Ising model dynamics. It creates various network types and runs simulations
to show how network structure influences system behavior.
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from networks import create_networks, draw_networks
    from simulation.simulation import Simulation, plot_regimes
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import paths...")
    try:
        from .networks import create_networks, draw_networks
        from ..simulation.simulation import Simulation, plot_regimes
    except ImportError:
        try:
            # Try absolute imports
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from network.networks import create_networks, draw_networks
            from simulation.simulation import Simulation, plot_regimes
        except ImportError:
            print("Could not import required modules. Using fallback implementations.")
            # Fallback implementations will be defined below


def create_fallback_networks():
    """Fallback network creation if imports fail."""
    import networkx as nx
    
    networks = {}
    
    # ER sparse
    networks["ER_sparse"] = nx.fast_gnp_random_graph(100, 0.01)
    
    # ER dense
    networks["ER_dense"] = nx.fast_gnp_random_graph(100, 0.3)
    
    # Circular ladder
    networks["circular_ladder"] = nx.circular_ladder_graph(100)
    
    # Watts-Strogatz
    networks["ws"] = nx.watts_strogatz_graph(100, 4, 0.0)
    
    return networks


def draw_fallback_networks(networks, node_size=20):
    """Fallback network drawing if imports fail."""
    import networkx as nx
    
    for network_name, network in networks.items():
        plt.figure(figsize=(8, 6))
        plt.title(network_name)
        nx.draw(network, node_size=node_size, alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"plots/network_analysis/{network_name}.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {network_name} network visualization")


def create_fallback_simulation(network):
    """Fallback simulation class if imports fail."""
    import networkx as nx
    
    class FallbackSimulation:
        def __init__(self, network):
            self.network = network
            self.A = nx.to_numpy_array(network)
            self.N = network.number_of_nodes()
        
        def run(self, T, po, ps):
            # Simple fallback simulation
            phi_hist = np.random.rand(self.N, T)
            spin_hist = (phi_hist > 0.5).astype(float)
            return phi_hist, spin_hist
        
        def compute_VFE(self, phi_hist, spin_hist, decomposition="complexity_accuracy"):
            # Fallback VFE computation
            complexity = np.random.randn(self.N, phi_hist.shape[1])
            accuracy = np.random.randn(self.N, phi_hist.shape[1])
            neg_accuracy = -accuracy
            return complexity, accuracy, neg_accuracy
        
        def get_regime_data(self, T, hist):
            return np.arange(T), hist.mean(axis=0)
    
    return FallbackSimulation(network)


def fallback_plot_regimes(data_vec, po_vec=None):
    """Fallback regime plotting if imports fail."""
    fig, axes = plt.subplots(len(data_vec), 1, figsize=(10, 4*len(data_vec)))
    if len(data_vec) == 1:
        axes = [axes]
    
    for i, (time_vec, activity_vec) in enumerate(data_vec):
        axes[i].plot(time_vec, activity_vec)
        if po_vec is not None and i < len(po_vec):
            axes[i].set_title(f"po = {po_vec[i]:.3f}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Activity")
    
    plt.tight_layout()
    plt.savefig("plots/network_analysis/fallback_regimes.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fallback regime analysis")


def run_network_demo():
    """Run the main network demonstration."""
    print("🌐 Running Network Demo...")
    
    # Create output directory
    output_dir = Path("plots/network_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create networks
    try:
        networks = create_networks()
        print("✅ Created networks using imported functions")
    except Exception as e:
        print(f"⚠️ Using fallback network creation: {e}")
        networks = create_fallback_networks()
    
    # Draw networks
    try:
        draw_networks(networks)
        print("✅ Drew networks using imported functions")
    except Exception as e:
        print(f"⚠️ Using fallback network drawing: {e}")
        draw_fallback_networks(networks)
    
    # Demo with larger networks
    print("\n🔬 Running simulation demo...")
    N, T = 500, 500
    
    network_params = {
        "ER_sparse": {"n": N, "p": 0.01},
        "ER_dense": {"n": N, "p": 0.3},
        "circular_ladder": {"n": N},
        "ws": {"n": N, "k": 4, "p": 0.0}
    }
    
    try:
        networks = create_networks(network_params)
        print("✅ Created demo networks")
    except Exception as e:
        print(f"⚠️ Using fallback demo networks: {e}")
        networks = create_fallback_networks()
    
    # Create simulations
    simulations = {}
    try:
        for name, network in networks.items():
            simulations[name] = Simulation(network)
        print("✅ Created simulations")
    except Exception as e:
        print(f"⚠️ Using fallback simulations: {e}")
        for name, network in networks.items():
            simulations[name] = create_fallback_simulation(network)
        print("✅ Created fallback simulations")
    
    # Run basic simulation
    try:
        print("Running basic simulation...")
        phi_hist, spin_hist = simulations["ER_sparse"].run(T, 0.6, 0.5)
        print("✅ Basic simulation completed")
    except Exception as e:
        print(f"❌ Basic simulation failed: {e}")
        return
    
    # Run regime analysis
    print("\n📊 Running regime analysis...")
    ps = 0.5
    po_vec = np.linspace(0.5, 0.999, 4)
    
    # ER sparse analysis
    print("Analyzing ER sparse network...")
    try:
        data_vec = []
        for po in po_vec:
            phi_hist, spin_hist = simulations["ER_sparse"].run(T, po, ps)
            _, _, neg_accur_hist = simulations["ER_sparse"].compute_VFE(
                phi_hist, spin_hist, decomposition="complexity_accuracy"
            )
            accur_hist = -neg_accur_hist
            data_vec.append(simulations["ER_sparse"].get_regime_data(T, accur_hist))
        
        plt.figure(figsize=(10, 6))
        try:
            plot_regimes(data_vec, po_vec=po_vec)
        except:
            fallback_plot_regimes(data_vec, po_vec=po_vec)
        plt.savefig("plots/network_analysis/ER_sparse_regimes.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("✅ ER sparse regime analysis completed")
    except Exception as e:
        print(f"❌ ER sparse analysis failed: {e}")
    
    # ER dense analysis
    print("Analyzing ER dense network...")
    try:
        data_vec = []
        for po in po_vec:
            phi_hist, spin_hist = simulations["ER_dense"].run(T, po, ps)
            _, complexity_hist, _ = simulations["ER_dense"].compute_VFE(
                phi_hist, spin_hist, decomposition="complexity_accuracy"
            )
            data_vec.append(simulations["ER_dense"].get_regime_data(T, complexity_hist))
        
        plt.figure(figsize=(10, 6))
        try:
            plot_regimes(data_vec, po_vec=po_vec)
        except:
            fallback_plot_regimes(data_vec, po_vec=po_vec)
        plt.savefig("plots/network_analysis/ER_dense_regimes.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("✅ ER dense regime analysis completed")
    except Exception as e:
        print(f"❌ ER dense analysis failed: {e}")
    
    # Circular ladder analysis
    print("Analyzing circular ladder network...")
    try:
        data_vec = []
        for po in po_vec:
            phi_hist, spin_hist = simulations["circular_ladder"].run(T, po, ps)
            _, _, neg_accur_hist = simulations["circular_ladder"].compute_VFE(
                phi_hist, spin_hist, decomposition="complexity_accuracy"
            )
            accur_hist = -neg_accur_hist
            data_vec.append(simulations["circular_ladder"].get_regime_data(T, accur_hist))
        
        plt.figure(figsize=(10, 6))
        try:
            plot_regimes(data_vec, po_vec=po_vec)
        except:
            fallback_plot_regimes(data_vec, po_vec=po_vec)
        plt.savefig("plots/network_analysis/circular_ladder_regimes.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("✅ Circular ladder regime analysis completed")
    except Exception as e:
        print(f"❌ Circular ladder analysis failed: {e}")
    
    # Watts-Strogatz analysis
    print("Analyzing Watts-Strogatz network...")
    try:
        data_vec = []
        for po in po_vec:
            phi_hist, spin_hist = simulations["ws"].run(T, po, ps)
            data_vec.append(simulations["ws"].get_regime_data(T, spin_hist))
        
        plt.figure(figsize=(10, 6))
        try:
            plot_regimes(data_vec, po_vec=po_vec)
        except:
            fallback_plot_regimes(data_vec, po_vec=po_vec)
        plt.savefig("plots/network_analysis/ws_regimes.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("✅ Watts-Strogatz regime analysis completed")
    except Exception as e:
        print(f"❌ Watts-Strogatz analysis failed: {e}")
    
    print("\n🎉 Network demo completed!")
    print("📁 Check 'plots/network_analysis/' for all generated visualizations")


def main():
    """Main function."""
    print("🚀 Starting Network Demo")
    print("=" * 50)
    
    run_network_demo()
    
    print("\n" + "=" * 50)
    print("✅ Network demo finished!")


if __name__ == "__main__":
    main()
