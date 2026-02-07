import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Central configuration for the HERCULES application.
    Loads settings from environment variables with sensible defaults.
    """
    
    # --- General Settings ---
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    PORT = int(os.getenv("PORT", 5000))
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./uploads")
    
    # --- LLM Configuration ---
    # Options: 'gemini', 'openai', 'anthropic', 'custom'
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    # API Keys (Security: Ensure these are set in .env, do not hardcode)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    # Custom LLM Settings (for vLLM / TGI / Local)
    CUSTOM_LLM_BASE_URL = os.getenv("CUSTOM_LLM_BASE_URL")
    CUSTOM_LLM_API_KEY = os.getenv("CUSTOM_LLM_API_KEY")
    CUSTOM_LLM_MODEL_NAME = os.getenv("CUSTOM_LLM_MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")

    # Model Names (Defaults)
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo-preview")
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-flash-latest")
    ANTHROPIC_MODEL_NAME = os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-opus-20240229")

    # --- Embedding Configuration ---
    # Options: 'openai', 'huggingface', 'gemini'
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    
    # HuggingFace Model (Local)
    # Using 'all-MiniLM-L6-v2' as requested for efficiency and performance
    HF_EMBEDDING_MODEL_NAME = os.getenv("HF_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    
    # OpenAI Embedding Model
    OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-3-small")

    # --- Clustering Configuration ---
    # Number of clusters to generate (can be dynamic, but this is a default/fallback)
    NUM_CLUSTERS = int(os.getenv("NUM_CLUSTERS", 5))
    
    # FAISS Index Type: 'Flat', 'IVF', 'HNSW'
    # 'Flat' is exact search (slower for huge data), 'IVF'/'HNSW' are approximate (faster)
    FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "Flat")

    # --- Summarization Configuration ---
    # Chunk size for splitting text
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

    @classmethod
    def validate(cls):
        """Validates critical configuration."""
        if cls.LLM_PROVIDER == 'openai' and not cls.OPENAI_API_KEY:
            raise ValueError("LLM_PROVIDER is 'openai' but OPENAI_API_KEY is missing.")
        if cls.LLM_PROVIDER == 'gemini' and not cls.GEMINI_API_KEY:
            raise ValueError("LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is missing.")
        if cls.LLM_PROVIDER == 'anthropic' and not cls.ANTHROPIC_API_KEY:
            raise ValueError("LLM_PROVIDER is 'anthropic' but ANTHROPIC_API_KEY is missing.")
        if cls.LLM_PROVIDER == 'custom' and not cls.CUSTOM_LLM_BASE_URL:
            raise ValueError("LLM_PROVIDER is 'custom' but CUSTOM_LLM_BASE_URL is missing.")

# Ensure upload directory exists
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
