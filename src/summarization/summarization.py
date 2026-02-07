from typing import Dict, Any
import time
from src.utils import chunk_text, setup_logger
from src.embeddings.embeddings import get_embedding_provider
from src.clustering.clustering import ClusterManager
from src.llm.factory import LLMFactory
from src.llm.prompts import Prompts
from src.config.config import Config

logger = setup_logger(__name__)

class SummarizationPipeline:
    """
    Orchestrates the HERCULES summarization process.
    """
    def __init__(self):
        self.embedding_provider = get_embedding_provider()
        self.llm_provider = LLMFactory.create_provider()
        self.cluster_manager = None 

    def run(self, text: str) -> Dict[str, Any]:
        """
        Runs the full pipeline on the input text.
        """
        start_total = time.time()
        
        # 1. Semantic Chunking
        logger.info("Starting semantic chunking...")
        
        # Try semantic chunking first, fallback to standard chunking
        try:
            from src.utils import semantic_chunking
            chunks = semantic_chunking(text, self.embedding_provider)
        except Exception as e:
            logger.warning(f"Semantic chunking failed ({e}), falling back to standard chunking.")
            chunks = chunk_text(text)

        if not chunks:
            raise ValueError("Text chunking resulted in empty list.")
        
        # 2. Embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_provider.embed_documents(chunks)
        dimension = embeddings.shape[1]
        
        # 3. Clustering
        logger.info("Clustering embeddings...")
        self.cluster_manager = ClusterManager(dimension=dimension)
        self.cluster_manager.create_index(embeddings)
        
        # Determine optimal clusters
        num_clusters = min(Config.NUM_CLUSTERS, len(chunks))
        centroids, assignments = self.cluster_manager.cluster_embeddings(embeddings, num_clusters=num_clusters)
        
        # 4. Representative Extraction
        logger.info("Extracting representative chunks...")
        representatives = self.cluster_manager.get_representative_chunks(embeddings, assignments, chunks)
        
        # 5. Cluster Summarization
        logger.info("Summarizing clusters...")
        cluster_summaries = []
        
        for cluster_id, rep_chunks in representatives.items():
            chunk_text_block = "\n---\n".join(rep_chunks)
            user_prompt = Prompts.CLUSTER_SUMMARY_USER_TEMPLATE.format(chunks=chunk_text_block)
            
            try:
                summary = self.llm_provider.generate(Prompts.CLUSTER_SUMMARY_SYSTEM, user_prompt)
                cluster_summaries.append(f"Cluster {cluster_id + 1}:\n{summary}")
            except Exception as e:
                logger.error(f"Failed to summarize cluster {cluster_id}: {e}")
                cluster_summaries.append(f"Cluster {cluster_id + 1}: [Error generating summary]")

        # 6. Final Summarization
        logger.info("Generating final summary...")
        all_cluster_summaries = "\n\n".join(cluster_summaries)
        
        final_user_prompt = Prompts.FINAL_SUMMARY_USER_TEMPLATE.format(cluster_summaries=all_cluster_summaries)
        final_summary = self.llm_provider.generate(Prompts.FINAL_SUMMARY_SYSTEM, final_user_prompt)
        
        duration = round(time.time() - start_total, 2)
        logger.info(f"Pipeline completed in {duration}s")
        
        return {
            "final_summary": final_summary,
            "cluster_summaries": cluster_summaries,
            "num_chunks": len(chunks),
            "num_clusters": num_clusters,
            "duration_seconds": duration
        }
