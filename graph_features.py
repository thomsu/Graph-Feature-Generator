import numpy as np
from numpy import linalg


class GraphInfo():

    def __init__(self):
        self._adj_matrix = None
        self._floyd_warshall = None
        self._node_degree = None
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

    def get_n_degree(self):
        if self._adj_matrix is None:
            print('No graph found. Please create graph by adding edges.')
            return
        if self._node_degree:
            return self._node_degree

        lookup = {v: k for k, v in self._node_dict.items()}
        self._node_degree = {}

        for i, col in enumerate(np.transpose(self._adj_matrix)):
            self._node_degree[lookup[i]] = sum(col)

        return self._node_degree

    def get_ev_centrality(self):
        eigval, eigvec = linalg.eig(self._adj_matrix)
        eigvec = eigvec[np.argmax(eigval)]

        self._eigenvector_centrality = {n: i for n, i in zip(sorted(self._node_dict), eigvec)}
        return self._eigenvector_centrality
