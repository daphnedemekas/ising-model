import os
from pathlib import Path
import numpy as np
import networkx as nx
from simulation import Simulation
from matplotlib import pyplot as plt
from math_utils import log_stable


def run():

    # n_x, n_y = 45, 45
    # G = nx.grid_2d_graph(n_x, n_y)
    T = 1000

    N = 500
    G = nx.fast_gnp_random_graph(N, p=0.04)
    N = G.number_of_nodes()
    A = nx.to_numpy_array(G)

    omega_init = 1.0

    ps = 0.5
    ps_vec = ps * np.ones(N)

    sim = Simulation(G, omega_matrix=np.ones((N, N)), p_s_vec=ps_vec)

    omega = 0.53
    phi_hist, spin_hist = sim.run(T, omega, ps)

    vfe, _, _ = sim.compute_VFE(phi_hist, spin_hist)

    hamiltonian = np.zeros(T)

    for t in range(1, T):
        hamiltonian[t] = sim.calculate_global_energy(spin_hist[:, t])

    until_T = T

    fig, ax = plt.subplots(2, 2, figsize=(14, 7), dpi=100)

    ax[0, 0].imshow(phi_hist[:, :until_T], aspect="auto", interpolation="none")
    ax[0, 0].set_title("Phi Hist")
    ax[0, 1].imshow(spin_hist[:, :until_T], aspect="auto", interpolation="none")
    ax[0, 1].set_title("Spin Hist")
    ax[1, 0].plot(vfe[:, :until_T].mean(axis=0))
    ax[1, 0].set_title("VFE")
    ax[1, 1].plot(hamiltonian[1:until_T])
    ax[1, 1].set_title("Total energy (H)")

    write_folder = "../"

    plt.savefig(os.path.join(write_folder, "tmp.png"), dpi=325, bbox_inches="tight")


if __name__ == "__main__":

    run()
