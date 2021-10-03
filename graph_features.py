import numpy as np


class GraphInfo():

    def __init__(self):
        self._adj_matrix = None
        self._floyd_warshall = None
        self._degree_centrality = None
        self._eigenvector_centrality = None
        self._betweenness_centrality = None
        self._closeness_centrality = None
        self._katz_index = None
        self._color_refinement = None

    def add_edges(self, edges):
        self._node_set = {i[0] for i in edges} | {i[1] for i in edges}
        self._node_dict = {n: i for i, n in enumerate(sorted(self._node_set))}
        size = len(self._node_set)
        self._adj_matrix = np.full((size, size), np.zeros(size))

        for start, end in edges:
            self._adj_matrix[self._node_dict[start]][self._node_dict[end]] += 1
            self._adj_matrix[self._node_dict[end]][self._node_dict[start]] += 1

    def degree_centrality(self):
        if self._adj_matrix is None:
            print('No graph found. Please create graph by adding edges.')
            return
        if self._degree_centrality:
            return self._degree_centrality

        lookup = {v: k for k, v in self._node_dict.items()}
        self._degree_centrality = {}

        for i, col in enumerate(np.transpose(self._adj_matrix)):
            self._degree_centrality[lookup[i]] = sum(col)

        return self._degree_centrality
