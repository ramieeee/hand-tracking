"""Small synchronous Ollama chat client for the hand-tracking project."""

from __future__ import annotations

import json
from urllib import error, request


class LLMChat:
    def __init__(
        self,
        model: str = "qwen3.5:2b",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        system_prompt: str = (
            "You interpret recognised hand-gesture meanings as concise commands. "
            "Do not invent gestures or actions that were not provided."
        ),
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.system_prompt = system_prompt

    def chat(self, message: str, *, system_prompt: str | None = None) -> str:
        """Send one message to Ollama and return the assistant's text."""
        if not message.strip():
            raise ValueError("message must not be empty")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt or self.system_prompt},
                {"role": "user", "content": message},
            ],
            "stream": False,
            "think": False,
            "options": {"num_ctx": 4096},
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                result = json.load(response)
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {details}") from exc
        except (error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. Check that 'ollama serve' is running."
            ) from exc

        return result.get("message", {}).get("content", "")

    def interpret_gesture(self, meaning: str) -> str:
        """Interpret an already-recognised gesture label; does not analyse landmarks."""
        return self.chat(
            f"Recognised gesture meaning: {meaning}\n"
            "Explain the intended command briefly. Do not claim to have executed it."
        )


if __name__ == "__main__":
    llm = LLMChat()
    print(llm.chat("Hello, introduce yourself in one sentence."))
