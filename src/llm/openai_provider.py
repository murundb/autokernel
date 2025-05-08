from openai import OpenAI

from .base import BaseLLMProvider

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key, model="gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.api_key = api_key
        self.model = model

    def chat(self, messages, system_prompt=None, max_tokens=2000, temperature=1):
        # OpenAI expects system prompt as the first message with role "system"
        openai_messages = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
        openai_messages.extend(messages)
        response = self.client.chat.completions.create(model=self.model,
                        messages=openai_messages,
                        # max_tokens=max_tokens,
                        temperature=temperature)
        return response.choices[0].message.content