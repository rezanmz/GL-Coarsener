from .embedding import Embedder
from .clustering import Cluster
from scipy.sparse import lil_matrix


class Coarsener:
    def __init__(self, adjacency_matrix):
        """
        Generate a coarser grid from a given fine adjacency matrix

        :param adjacency_matrix: scipy sparse CRS
        """
        self.adjacency_matrix = adjacency_matrix

    def apply(
        self,
        dimensions,
        walk_length,
        num_walks,
        p,
        q,
        number_of_clusters=None,
        clustering_method='kmeans',
        workers=1
    ):
        """
        Generates coarse level adjacency matrix using GL-Coarsener method, as proposed in https://arxiv.org/abs/2011.09994

        :param dimensions: `Embedding space dimensions`
        :param walk_length: Length of random-walk
        :param num_walks: Number of random-walks performed
        :param p: Return parameter
        :param q: In-Out parameter
        :param number_of_clusters: Size of the coarse matrix, default: adjacency_matrix.shape[0] // 5
        :param clustering_method: Clustering method to be used; could be 'kmeans' of 'minibatch_kmeans, default: 'kmeans'
        :param workers: Number of workers available for parallel computation, default: 1

        :rtype: scipy sparse CRS of size (number_of_clusters, number_of_clusters)
        """
        # Initial value for number of clusters
        if number_of_clusters is None:
            number_of_clusters = self.adjacency_matrix.shape[0] // 5

        # Step 1: Generate embeddin vectors from the given adjacency matrix (generate embedding space)
        embedder = Embedder(adjacency_matrix=self.adjacency_matrix)
        embedding_vectors = embedder.node2vec(
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
            p=p,
            q=q
        )

        # Step 2: Feed the embedding vectors to clustering module and generate clusters of nodes
        cluster = Cluster(embedding_vectors=embedding_vectors)
        if clustering_method == 'kmeans':
            groups = cluster.kmeans(number_of_clusters=number_of_clusters)
        elif clustering_method == 'minibatch_kmeans':
            groups = cluster.minibatch_kmeans(
                number_of_clusters=number_of_clusters)

        # Step 3: Compute prolongation and restriction operators
        # Manipulating LIL sparse matrix is more efficient than CSR sparse matrix
        p = lil_matrix((self.adjacency_matrix.shape[0], number_of_clusters))
        for i in range(self.adjacency_matrix.shape[0]):
            for j in range(len(groups)):
                p[i, j] = 1 if i in groups[j] else 0
        # Performing matrix operations on CSR sparse matrix is more efficient than LIL sparse matrix
        p = p.tocsr()

        return p
