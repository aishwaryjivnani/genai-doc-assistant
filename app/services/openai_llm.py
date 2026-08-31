"""Small LangChain-compatible adapter for the OpenAI Responses API."""
from typing import Any, Iterable

from langchain_core.messages import AIMessage
from openai import OpenAI


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Iterable):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)


class OpenAIResponsesChatModel:
    """Expose ``invoke(messages)`` for the existing agent graph."""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def invoke(self, messages) -> AIMessage:
        instructions = []
        inputs = []
        for message in messages:
            text = _content_to_text(message.content)
            if getattr(message, "type", "") == "system":
                instructions.append(text)
            else:
                role = "assistant" if getattr(message, "type", "") == "ai" else "user"
                inputs.append({"role": role, "content": text})

        response = self.client.responses.create(
            model=self.model,
            instructions="\n\n".join(instructions) or None,
            input=inputs,
        )
        return AIMessage(content=response.output_text.strip())
