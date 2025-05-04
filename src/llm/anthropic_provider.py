import anthropic
from .base import BaseLLMProvider

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)

    def chat(self, messages, system_prompt=None, max_tokens=2000, temperature=1):
        response = self.client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text