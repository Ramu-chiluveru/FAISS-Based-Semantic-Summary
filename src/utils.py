import logging
import json
import time
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.config import Config

class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    Essential for production monitoring and scalability.
    """
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        if hasattr(record, "props"):
            log_record.update(record.props)
        return json.dumps(log_record)

def setup_logger(name: str) -> logging.Logger:
    """
    Configures a logger with JSON formatting.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO if not Config.DEBUG else logging.DEBUG)
    
    # Prevent adding multiple handlers if function is called repeatedly
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
    
    return logger

logger = setup_logger(__name__)

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Splits text into chunks using standard recursive character splitting.
    Fallback for when semantic chunking fails or is not needed.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    return splitter.split_text(text)

def semantic_chunking(text: str, embedding_provider, threshold: float = 0.8, min_chunk_size: int = 500) -> List[str]:
    """
    Splits text into semantically coherent chunks based on embedding similarity.
    Ref: 'LLM-Enhanced Semantic Text Segmentation'
    
    Args:
        text: Full document text.
        embedding_provider: Instance of BaseEmbedding to generate vectors.
        threshold: Similarity threshold to trigger a split (lower = more splits).
        min_chunk_size: Minimum characters per chunk to avoid tiny fragments.
    """
    import numpy as np
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    start_time = time.time()
    
    # 1. Initial split into sentences (or small blocks)
    # We use a small chunk size to get granular blocks for comparison
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    sentences = splitter.split_text(text)
    
    if len(sentences) < 2:
        return sentences
        
    # 2. Embed all sentences
    # Note: This increases API calls. For production, batching is crucial.
    embeddings = embedding_provider.embed_documents(sentences)
    
    # 3. Calculate cosine similarity between adjacent sentences
    # Norms for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1)
    # Avoid division by zero
    norms[norms == 0] = 1e-10
    normalized_embeddings = embeddings / norms[:, np.newaxis]
    
    similarities = np.diag(np.dot(normalized_embeddings[:-1], normalized_embeddings[1:].T))
    
    # 4. Group sentences into chunks based on similarity dips
    chunks = []
    current_chunk = [sentences[0]]
    current_len = len(sentences[0])
    
    for i, sim in enumerate(similarities):
        next_sentence = sentences[i+1]
        
        # If similarity drops below threshold AND we have enough content, split.
        # OR if the current chunk is getting too large (fallback).
        if (sim < threshold and current_len > min_chunk_size) or (current_len > Config.CHUNK_SIZE):
            chunks.append(" ".join(current_chunk))
            current_chunk = [next_sentence]
            current_len = len(next_sentence)
        else:
            current_chunk.append(next_sentence)
            current_len += len(next_sentence)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    logger.info(f"Semantic chunking complete", extra={"props": {
        "original_sentences": len(sentences),
        "final_chunks": len(chunks),
        "duration_seconds": round(time.time() - start_time, 4)
    }})
    
    return chunks
