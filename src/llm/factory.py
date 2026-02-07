from src.config.config import Config
from src.llm.providers import LLMProvider, OpenAIProvider, GeminiProvider, AnthropicProvider, CustomProvider
from src.utils import setup_logger

logger = setup_logger(__name__)

class LLMFactory:
    """
    Factory to create LLM providers based on configuration.
    """
    @staticmethod
    def create_provider() -> LLMProvider:
        provider_name = Config.LLM_PROVIDER
        logger.info(f"Initializing LLM Provider: {provider_name}")
        
        if provider_name == 'openai':
            return OpenAIProvider()
        elif provider_name == 'gemini':
            return GeminiProvider()
        elif provider_name == 'anthropic':
            return AnthropicProvider()
        elif provider_name == 'custom':
            return CustomProvider()
        else:
            logger.warning(f"Unknown provider '{provider_name}', defaulting to Gemini")
            return GeminiProvider()
