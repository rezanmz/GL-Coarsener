import networkx
from node2vec import Node2Vec
import numpy as np
import time


class Embedder:
    def __init__(self, adjacency_matrix=None, graph=None):
        """
        Embed graph to an n-dimensional embedding space
        A graph or an adjacency matrix should be provided. If graph is provided, adjacency matrix will be ignored.

        :param adjacency_matrix: scipy sparse CSR
        :param graph: networkx graph
        """
        if graph:
            self.graph = graph
        else:
            self.graph = networkx.from_scipy_sparse_matrix(
                abs(adjacency_matrix))
        if graph is None and adjacency_matrix is None:
            raise Exception('Graph or AdjacencyMatrix should be provided.')

    def node2vec(self, dimensions, walk_length, num_walks, workers=1, p=1, q=1):
        """
        Generates embedding space from the given graph/adjacency_matrix in a way that similar nodes in the graph are close to each other in the embedding space.
        For more information, visit https://snap.stanford.edu/node2vec/

        :param dimensions: Embedding space dimensions
        :param walk_length: Length of random-walk
        :param num_walks: Number of random-walks performed
        :param workers: Number of workers available for parallel computation, default: 1
        :param p: Return parameter, default: 1
        :param q: In-Out parameter, default: 1

        :return: embedding space as a numpy array of shape (number_of_nodes, dimensions)
        """
        init_time = time.time()
        model = Node2Vec(
            self.graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
            p=p,
            q=q
        )
        # Fit node2vec model
        model = model.fit()

        # Extract embedding vectors for each node
        embedding_vectors = np.zeros(
            (self.graph.number_of_nodes(), dimensions))
        for i in range(self.graph.number_of_nodes()):
            embedding_vectors[i] = model.wv.get_vector(str(i))

        print(
            f'Elapsed time (node2vec embedding): {(time.time() - init_time):.2f}s')

        return embedding_vectors
