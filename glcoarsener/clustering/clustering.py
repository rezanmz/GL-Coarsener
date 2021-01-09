from sklearn.cluster import KMeans, MiniBatchKMeans
import time
import numpy as np


class Cluster:
    def __init__(self, embedding_vectors):
        """
        An embeddding space (embedding vectors) should be provided
        """
        self.embedding_vectors = embedding_vectors

    def kmeans(self, number_of_clusters):
        """
        Cluster nodes in the embedding space using KMeans clustering algorithm

        :param number_of_clusters: Number of clusters to be generated
        :return: Generated clusters
        """
        init_time = time.time()
        clusters = KMeans(
            init='random',
            n_clusters=number_of_clusters,
            verbose=1
        ).fit(self.embedding_vectors)

        # Generate a list of clustered nodes
        groups = [[] for _ in range(max(clusters.labels_) + 1)]
        for node_index, group_index in enumerate(clusters.labels_):
            groups[group_index].append(node_index)

        groups = [group for group in groups if len(group) != 0]
        result = []
        for index, group in enumerate(groups):
            for node in group:
                result.append((node, index))

        print(
            f'Elapsed time (KMeans clustering): {(time.time() - init_time):.2f}s')
        return np.array(result)

    def minibatch_kmeans(self, number_of_clusters):
        """
        Cluster nodes in the embedding space using Mini-Batch KMeans clustering algorithm

        number_of_clusters: Number of clusters to be generated
        :return: Generated clusters
        """
        init_time = time.time()
        clusters = MiniBatchKMeans(
            init='random',
            init_size=number_of_clusters * 5,
            n_clusters=number_of_clusters,
            verbose=1,
            batch_size=(number_of_clusters // 20) + 1,
            reassignment_ratio=1,
            max_iter=500,
            n_init=2,
            max_no_improvement=50
        ).fit(self.embedding_vectors)

        # Generate a list of clustered nodes
        groups = [[] for _ in range(max(clusters.labels_) + 1)]
        for node_index, group_index in enumerate(clusters.labels_):
            groups[group_index].append(node_index)

        groups = [group for group in groups if len(group) != 0]
        result = []
        for index, group in enumerate(groups):
            for node in group:
                result.append((node, index))
        print(
            f'Elapsed time (Mini-Batch KMeans clustering): {(time.time() - init_time):.2f}s')
        return np.array(result)
