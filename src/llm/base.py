from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    @abstractmethod
    def chat(self, messages, system_prompt=None, max_tokens=2000, temperature=1):
        pass