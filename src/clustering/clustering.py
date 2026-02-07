import numpy as np
import faiss
from typing import List, Tuple, Dict
from src.config.config import Config
from src.utils import setup_logger

logger = setup_logger(__name__)

class ClusterManager:
    """
    Manages clustering of text embeddings using FAISS.
    Scalable to millions of vectors using IVF indices.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = None
        self.kmeans = None
        self.centroids = None

    def create_index(self, embeddings: np.ndarray):
        """
        Creates a FAISS index for the embeddings.
        """
        try:
            index_type = Config.FAISS_INDEX_TYPE
            
            if index_type == 'Flat':
                self.index = faiss.IndexFlatL2(self.dimension)
            elif index_type == 'IVF':
                # IVF requires training
                nlist = min(100, int(np.sqrt(len(embeddings))))  # Heuristic for nlist
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                self.index.train(embeddings)
            else:
                # Default to Flat if unknown
                self.index = faiss.IndexFlatL2(self.dimension)
                
            self.index.add(embeddings)
            logger.info(f"FAISS index created with {self.index.ntotal} vectors. Type: {index_type}")
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            raise

    def cluster_embeddings(self, embeddings: np.ndarray, num_clusters: int = Config.NUM_CLUSTERS) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs K-means clustering on the embeddings.
        
        FIRST PRINCIPLES:
        1. **Vector Space**: We converted text into 384-dimensional vectors (arrows in space).
           Similar texts point in similar directions.
        2. **Clustering**: We want to group these arrows into 'k' bundles.
        3. **K-Means Algorithm**:
           - Step A: Pick 'k' random points as "Centroids".
           - Step B: Assign every text vector to its nearest Centroid.
           - Step C: Move the Centroid to the *average* position of its assigned vectors.
           - Repeat B & C until they stop moving.
           
        The result? 'k' groups of semantically related text chunks.
        """
        try:
            # FAISS Implementation
            # FAISS is optimized for this. It runs K-Means much faster than Scikit-Learn
            # by using highly optimized C++ kernels (and GPU if available).
            
            # niter=20: How many times to repeat Step B & C.
            # nredo=1: If a cluster becomes empty, restart with a new seed.
            kmeans = faiss.Kmeans(d=self.dimension, k=num_clusters, niter=20, verbose=False)
            kmeans.train(embeddings)
            
            self.centroids = kmeans.centroids
            
            # ASSIGNMENT STEP:
            # Now that we found the best centroids, we need to ask:
            # "For each text vector, which centroid is closest?"
            # We do this by searching the index of centroids.
            
            centroid_index = faiss.IndexFlatL2(self.dimension)
            centroid_index.add(self.centroids)
            
            # search() returns:
            # D: Distances to the nearest neighbor
            # I: Index (ID) of the nearest neighbor (this is our cluster assignment!)
            _, assignments = centroid_index.search(embeddings, 1)
            
            return self.centroids, assignments.flatten()
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            raise

    def get_representative_chunks(self, embeddings: np.ndarray, assignments: np.ndarray, chunks: List[str]) -> Dict[int, List[str]]:
        """
        Finds the chunks closest to each cluster centroid.
        Returns a dictionary mapping cluster ID to a list of representative chunks.
        """
        representatives = {}
        unique_clusters = np.unique(assignments)
        
        for cluster_id in unique_clusters:
            # Get indices of points in this cluster
            cluster_indices = np.where(assignments == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            cluster_embeddings = embeddings[cluster_indices]
            centroid = self.centroids[cluster_id].reshape(1, -1)
            
            # Find the point closest to the centroid in this cluster
            # We can use a temporary index or just simple distance
            temp_index = faiss.IndexFlatL2(self.dimension)
            temp_index.add(cluster_embeddings)
            
            # Find top 3 representative chunks for this cluster
            k_rep = min(3, len(cluster_indices))
            _, relative_indices = temp_index.search(centroid, k_rep)
            
            # Map back to original indices
            original_indices = cluster_indices[relative_indices.flatten()]
            
            representatives[int(cluster_id)] = [chunks[idx] for idx in original_indices]
            
        return representatives
