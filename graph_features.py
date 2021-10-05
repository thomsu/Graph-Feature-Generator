import numpy as np
from numpy import linalg
from collections import Counter


class GraphInfo():

    def __init__(self):
        self._adj_matrix = None
        self._shortest_paths = None
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
            self._adj_matrix[self._node_dict[start]][self._node_dict[end]] = 1
            self._adj_matrix[self._node_dict[end]][self._node_dict[start]] = 1

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

    def floyd_warshall(self):
        size = len(self._node_set)
        node_list = list(sorted(self._node_set))
        dp = [[self._adj_matrix[i][j] if i == j or self._adj_matrix[i][j] >
               0 else float('inf') for j in range(size)] for i in range(size)]
        next_node = [[node_list[j] if dp[i][j] != float(
            'inf') else None for j in range(size)] for i in range(size)]

        for k in range(size):
            for i in range(size):
                for j in range(size):
                    if dp[i][k] + dp[k][j] < dp[i][j]:
                        dp[i][j] = dp[i][k] + dp[k][j]
                        next_node[i][j] = next_node[i][k]

        self._shortest_paths = {}
        for i in range(size-1):
            for j in range(i+1, size):
                path = []
                if dp[i][j] == float('inf'):
                    self._shortest_paths[f'{node_list[i]}-{node_list[j]}'] = None
                    self._shortest_paths[f'{node_list[j]}-{node_list[i]}'] = None

                curr = node_list[i]
                while curr is not None and curr != node_list[j]:
                    if curr is not None:
                        path.append(curr)
                    curr = next_node[node_list.index(curr)][j]
                if curr is not None:
                    path.append(curr)
                self._shortest_paths[f'{node_list[i]}-{node_list[j]}'] = path
                self._shortest_paths[f'{node_list[j]}-{node_list[i]}'] = path[::-1]

    def get_ev_centrality(self):
        if self._eigenvector_centrality is not None:
            return self._eigenvector_centrality

        eigval, eigvec = linalg.eig(self._adj_matrix)
        eigvec = eigvec[np.argmax(eigval)]

        self._eigenvector_centrality = {n: i for n, i in zip(sorted(self._node_dict), eigvec)}
        return self._eigenvector_centrality

    def get_btwn_centrality(self):
        if self._betweenness_centrality is not None:
            return self._betweenness_centrality

        size = len(self._node_set)
        node_list = list(sorted(self._node_set))

        if self._shortest_paths is None:
            self.floyd_warshall()

        self._betweenness_centrality = {}

        for k in range(size):
            count = 0
            total = 0
            for i in range(size):
                for j in range(size):
                    if k == i or k == j or i >= j:
                        continue
                    total += 1
                    if node_list[k] in self._shortest_paths[f'{node_list[i]}-{node_list[j]}']:
                        count += 1
            self._betweenness_centrality[node_list[k]] = count / total

        return self._betweenness_centrality

    def get_close_centrality(self):
        if self._closeness_centrality is not None:
            return self._closeness_centrality

        size = len(self._node_set)
        node_list = list(sorted(self._node_set))

        if self._shortest_paths is None:
            self.floyd_warshall()

        self._closeness_centrality = {}

        for i in range(size):
            length = 0
            for j in range(size):
                if i == j:
                    continue
                if self._shortest_paths[f'{node_list[i]}-{node_list[j]}'] is None:
                    self._closeness_centrality[node_list[i]] = 0
                    break
                length += len(self._shortest_paths[f'{node_list[i]}-{node_list[j]}']) - 1
            else:
                self._closeness_centrality[node_list[i]] = 1 / length

        return self._closeness_centrality

    def get_katz_index(self, beta=None):
        if beta is None:
            beta = 0.7

        if beta <= 0 or beta >= 1:
            print('beta should be between 0 and 1 exclusive.')
            return

        size = len(self._node_set)
        node_list = list(sorted(self._node_set))

        katz_matrix = linalg.inv(np.identity(size)-beta*self._adj_matrix) - np.identity(size)
        self._katz_index = {}

        for i in range(size):
            for j in range(size):
                self._katz_index[f'{node_list[i]}-{node_list[j]}'] = katz_matrix[i][j]

        return self._katz_index

    def get_col_refinement(self, step=None):
        if step is None:
            step = 3

        if step < 2:
            print('step should be an integer greater than 1.')
            return

        size = len(self._node_set)
        adj_list = [[] for _ in range(size)]

        for j in range(size):
            for i in range(size):
                if self._adj_matrix[i][j] > 0:
                    adj_list[j].append(i)

        prev, k = [1] * size, 1
        self._color_refinement = Counter()
        self._color_refinement[(0, 1)] += size

        while k < step:
            curr = [0] * size
            for node in range(size):
                for neigbr in adj_list[node]:
                    curr[node] += prev[neigbr]

            for i in range(size):
                self._color_refinement[(prev[i], curr[i])] += 1
            prev = curr
            k += 1

        return self._color_refinement
