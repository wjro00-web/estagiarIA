"""
chat_controller.py
Gerencia o estado da conversa e orquestra o fluxo entre UI e serviço.
Princípio SOLID aplicado: Open/Closed Principle (OCP) — aberto para extensão,
fechado para modificação. Novo comportamento via composição, não herança.
"""

from src.claude_service import ClaudeService, ChatMessage, ChatResponse


SYSTEM_PROMPT = (
    "Você é um assistente prestativo, claro e direto. "
    "Responda sempre em português do Brasil, de forma concisa e precisa."
)


class ChatController:
    """
    Orquestra a conversa entre o usuário e o ClaudeService.
    Mantém o histórico da sessão e delega o envio ao serviço.
    """

    def __init__(self, service: ClaudeService):
        self._service = service
        self._history: list[ChatMessage] = []

    @property
    def history(self) -> list[ChatMessage]:
        """Retorna uma cópia imutável do histórico."""
        return list(self._history)

    def send(self, user_message: str) -> ChatResponse:
        """
        Processa a mensagem do usuário, atualiza o histórico e retorna a resposta.

        Args:
            user_message: Texto enviado pelo usuário.

        Returns:
            ChatResponse com a resposta do assistente.
        """
        self._history.append(ChatMessage(role="user", content=user_message))

        response = self._service.send_conversation(
            messages=self._history,
            system_prompt=SYSTEM_PROMPT,
        )

        self._history.append(ChatMessage(role="assistant", content=response.content))

        return response

    def clear_history(self) -> None:
        """Limpa o histórico da conversa atual."""
        self._history = []
