import networkx as nx
import random
import matplotlib.pyplot as plt
from .config import graph_generation_fns as gen_funcs


def create_network_dict(
    network_params={
        "ER": {"n": 100, "p": 0.01},
        "circular_ladder": {"n": 100},
        "grid": {"dim": [10, 10]},
        "ws": {"n": 100, "k": 4, "p": 0.0},
    }
):
    """
    Create a dictionary containing networkx graph objects, whose class and parameterisation is stored in an input dictionary `network_params`
    """

    network_dict = {
        network_name: gen_funcs[network_name](**params)
        for network_name, params in network_params.items()
    }

    return network_dict


def create_networks(
    network_params={
        "ER_sparse": {"n": 100, "p": 0.01},
        "ER_dense": {"n": 100, "p": 0.3},
        "circular_ladder": {"n": 100},
        "grid": {"dim": [10, 10]},
        "ws": {"n": 100, "k": 4, "p": 0.0},
    }
):
    networks = {}
    network_names = network_params.keys()
    if "ER_sparse" in network_names:
        params = network_params["ER_sparse"]
        ER_sparse = nx.fast_gnp_random_graph(**params)
        networks["ER_sparse"] = ER_sparse
    if "ER_dense" in network_names:
        params = network_params["ER_dense"]
        ER_dense = nx.fast_gnp_random_graph(**params)
        networks["ER_dense"] = ER_dense
    if "circular_ladder" in network_names:
        params = network_params["circular_ladder"]
        circular_ladder = nx.circular_ladder_graph(**params)
        networks["circular_ladder"] = circular_ladder
    if "grid" in network_names:
        params = network_params["grid"]
        grid = nx.grid_graph(**params)
        networks["grid"] = grid
    if "ws" in network_names:
        params = network_params["ws"]
        ws = nx.watts_strogatz_graph(**params)
        networks["ws"] = ws

    return networks


def draw_networks(networks, node_size=20):
    for network_name, network in networks.items():
        plt.figure()
        plt.title(network_name)
        nx.draw(network, node_size=node_size)
        plt.show()
