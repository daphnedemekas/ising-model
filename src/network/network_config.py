import networkx as nx

# can add more graphs to the look-up table here
graph_generation_fns = {
    "ER": nx.fast_gnp_random_graph,
    "circular_ladder": nx.circular_ladder_graph,
    "grid": nx.grid_graph,
    "ws": nx.watts_strogatz_graph,
}
