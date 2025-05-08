import json

class PromptHistory:
    def __init__(self, history=None):
        self.history = history or []

    def add(self, role, content):
        self.history.append({"role": role, "content": content})

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def trim_to_fit(self, max_tokens, tokenizer):
        # Remove oldest messages until under max_tokens
        while tokenizer(self.history) > max_tokens and len(self.history) > 1:
            self.history.pop(0)