"""
claude_service.py
Responsável exclusivamente pela comunicação com a API da Anthropic (Claude).
Princípio SOLID aplicado: Single Responsibility Principle (SRP).
"""

import os
import anthropic
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """Representa uma mensagem do chat."""
    role: str
    content: str


@dataclass
class ChatResponse:
    """Representa a resposta recebida da API."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int


class ClaudeService:
    """
    Serviço de comunicação com a API da Anthropic.
    Segue o princípio de responsabilidade única (SRP):
    apenas gerencia o envio e recebimento de mensagens.
    """

    DEFAULT_MODEL = "claude-sonnet-4-5"
    DEFAULT_MAX_TOKENS = 1024

    def __init__(self, api_key: str = None, model: str = None):
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError("ANTHROPIC_API_KEY não encontrada nas variáveis de ambiente.")

        self._client = anthropic.Anthropic(api_key=resolved_key)
        self._model = model or self.DEFAULT_MODEL

    def send_message(self, user_message: str, system_prompt: str = None) -> ChatResponse:
        """
        Envia uma mensagem para o Claude e retorna a resposta.

        Args:
            user_message: Mensagem do usuário.
            system_prompt: Instrução de sistema opcional.

        Returns:
            ChatResponse com o conteúdo da resposta e metadados.
        """
        if not user_message or not user_message.strip():
            raise ValueError("A mensagem não pode ser vazia.")

        kwargs = {
            "model": self._model,
            "max_tokens": self.DEFAULT_MAX_TOKENS,
            "messages": [{"role": "user", "content": user_message}],
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)

        return ChatResponse(
            content=response.content[0].text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def send_conversation(self, messages: list[ChatMessage], system_prompt: str = None) -> ChatResponse:
        """
        Envia uma conversa multi-turno para o Claude.

        Args:
            messages: Lista de mensagens anteriores.
            system_prompt: Instrução de sistema opcional.

        Returns:
            ChatResponse com a resposta do modelo.
        """
        if not messages:
            raise ValueError("A lista de mensagens não pode ser vazia.")

        formatted = [{"role": msg.role, "content": msg.content} for msg in messages]

        kwargs = {
            "model": self._model,
            "max_tokens": self.DEFAULT_MAX_TOKENS,
            "messages": formatted,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)

        return ChatResponse(
            content=response.content[0].text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
