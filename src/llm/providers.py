from abc import ABC, abstractmethod
from src.config.config import Config
from src.utils import setup_logger

logger = setup_logger(__name__)

class LLMProvider(ABC):
    """
    Abstract Strategy for LLM Providers.
    """
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generates a response from the LLM.
        """
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            self.model = Config.OPENAI_MODEL_NAME
        except ImportError:
            logger.error("openai package missing.")
            raise

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

class GeminiProvider(LLMProvider):
    def __init__(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model_name = Config.GEMINI_MODEL_NAME
            # We don't instantiate self.model here because we need dynamic system_instruction
        except ImportError:
            logger.error("google-generativeai package missing.")
            raise

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import google.generativeai as genai
            # Instantiate model with system_instruction for each call to support dynamic system prompts
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt
            )
            response = model.generate_content(user_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise

class AnthropicProvider(LLMProvider):
    def __init__(self):
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
            self.model = Config.ANTHROPIC_MODEL_NAME
        except ImportError:
            logger.error("anthropic package missing.")
            raise

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4096
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise

class CustomProvider(LLMProvider):
    """
    Generic provider for OpenAI-compatible endpoints (vLLM, TGI, Local).
    """
    def __init__(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=Config.CUSTOM_LLM_BASE_URL,
                api_key=Config.CUSTOM_LLM_API_KEY or "dummy"
            )
            self.model = Config.CUSTOM_LLM_MODEL_NAME
        except ImportError:
            logger.error("openai package missing (needed for custom provider).")
            raise

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Custom LLM generation failed: {e}")
            raise
