from abc import ABC, abstractmethod
from typing import List
import numpy as np
from src.config.config import Config
from src.utils import setup_logger

logger = setup_logger(__name__)

class BaseEmbedding(ABC):
    """
    Abstract base class for embedding providers.
    Enables swapping between OpenAI, HuggingFace, Gemini, etc.
    """
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of texts into a numpy array of vectors.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """
        Embeds a single query text.
        """
        pass

class OpenAIEmbedding(BaseEmbedding):
    def __init__(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            self.model = Config.OPENAI_EMBEDDING_MODEL_NAME
            logger.info(f"Initialized OpenAIEmbedding with model: {self.model}")
        except ImportError:
            logger.error("openai package not found. Please install it.")
            raise

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        try:
            # OpenAI batch limit handling could be added here for very large lists
            response = self.client.embeddings.create(input=texts, model=self.model)
            embeddings = [data.embedding for data in response.data]
            return np.array(embeddings, dtype='float32')
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

class HuggingFaceEmbedding(BaseEmbedding):
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model_name = Config.HF_EMBEDDING_MODEL_NAME
            logger.info(f"Loading HuggingFace model: {self.model_name} on {device}...")
            self.model = SentenceTransformer(self.model_name, device=device)
            logger.info("HuggingFace model loaded successfully.")
        except ImportError:
            logger.error("sentence-transformers or torch not found.")
            raise

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.astype('float32')
        except Exception as e:
            logger.error(f"HuggingFace embedding failed: {e}")
            raise

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

class GeminiEmbedding(BaseEmbedding):
    def __init__(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = "models/embedding-001" # Standard Gemini embedding model
            logger.info(f"Initialized GeminiEmbedding with model: {self.model}")
        except ImportError:
            logger.error("google-generativeai package not found.")
            raise

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        try:
            import google.generativeai as genai
            # Gemini batch embedding might require iteration if list is huge
            # Current API supports batch_embed_contents
            result = genai.embed_content(
                model=self.model,
                content=texts,
                task_type="retrieval_document"
            )
            # Check response structure - it returns a dict with 'embedding'
            # For list input, it might return a list of embeddings
            if 'embedding' in result:
                 return np.array([result['embedding']], dtype='float32')
            # If batch is supported natively in a list, handle it. 
            # Note: genai.embed_content usually takes a single string or list.
            # Let's assume list support or iterate.
            # Actually, for safety and API limits, let's iterate or use batch methods if available.
            # For MVP, simple iteration is safer to avoid API complexity.
            embeddings = []
            for text in texts:
                res = genai.embed_content(model=self.model, content=text, task_type="retrieval_document")
                embeddings.append(res['embedding'])
            return np.array(embeddings, dtype='float32')
        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            raise

    def embed_query(self, text: str) -> np.ndarray:
        try:
            import google.generativeai as genai
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_query"
            )
            return np.array(result['embedding'], dtype='float32')
        except Exception as e:
            logger.error(f"Gemini query embedding failed: {e}")
            raise

def get_embedding_provider() -> BaseEmbedding:
    """Factory to get the configured embedding provider."""
    provider = Config.EMBEDDING_PROVIDER
    if provider == 'openai':
        return OpenAIEmbedding()
    elif provider == 'huggingface':
        return HuggingFaceEmbedding()
    elif provider == 'gemini':
        return GeminiEmbedding()
    else:
        logger.warning(f"Unknown provider {provider}, defaulting to HuggingFace")
        return HuggingFaceEmbedding()
