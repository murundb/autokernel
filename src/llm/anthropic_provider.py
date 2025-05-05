import anthropic
from .base import BaseLLMProvider

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)

    def chat(self, messages, system_prompt=None, max_tokens=20000, temperature=1):
        think = True
        if think:
            response = self.client.messages.create(
                model="claude-3-7-sonnet-latest",
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=messages,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 4000
                }
            )
            if len(response.content) > 1:
                return response.content[-1].text
            else:
                return None
        else:
            response = self.client.messages.create(
                model="claude-3-7-sonnet-latest",
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=messages
            )
            return response.content[0].text