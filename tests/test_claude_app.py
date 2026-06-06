"""
test_claude_app.py
Testes unitários para validação de envio e recebimento da API do Claude.
Utiliza mocks para isolar as unidades lógicas sem dependência de rede.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.claude_service import ClaudeService, ChatMessage, ChatResponse
from src.chat_controller import ChatController


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_mock_response(text: str = "Olá! Como posso ajudar?") -> MagicMock:
    """Cria um mock da resposta da API da Anthropic."""
    mock_content = MagicMock()
    mock_content.text = text

    mock_usage = MagicMock()
    mock_usage.input_tokens = 10
    mock_usage.output_tokens = 20

    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_response.model = "claude-sonnet-4-20250514"
    mock_response.usage = mock_usage

    return mock_response


# ---------------------------------------------------------------------------
# Teste 1: Validação de ENVIO
# Garante que a mensagem do usuário é corretamente formatada e enviada.
# ---------------------------------------------------------------------------

class TestEnvioMensagem:
    """Testes para validar a unidade lógica de envio de mensagens."""

    @patch("src.claude_service.anthropic.Anthropic")
    def test_envia_mensagem_simples_com_conteudo_correto(self, MockAnthropic):
        """
        Verifica que a mensagem enviada à API contém o texto correto
        no formato esperado pela biblioteca Anthropic.
        """
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = make_mock_response()

        service = ClaudeService(api_key="fake-key-for-test")
        service.send_message("Qual é a capital do Brasil?")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        messages_sent = call_kwargs["messages"]

        assert len(messages_sent) == 1
        assert messages_sent[0]["role"] == "user"
        assert messages_sent[0]["content"] == "Qual é a capital do Brasil?"

    @patch("src.claude_service.anthropic.Anthropic")
    def test_nao_envia_mensagem_vazia(self, MockAnthropic):
        """
        Garante que uma mensagem vazia levanta ValueError antes
        de qualquer chamada à API.
        """
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client

        service = ClaudeService(api_key="fake-key-for-test")

        with pytest.raises(ValueError, match="mensagem não pode ser vazia"):
            service.send_message("   ")

        mock_client.messages.create.assert_not_called()

    @patch("src.claude_service.anthropic.Anthropic")
    def test_controller_adiciona_mensagem_ao_historico_antes_de_enviar(self, MockAnthropic):
        """
        Verifica que o ChatController registra a mensagem do usuário
        no histórico antes de enviar à API.
        """
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = make_mock_response()

        service = ClaudeService(api_key="fake-key-for-test")
        controller = ChatController(service)
        controller.send("Me explique o que é CI/CD.")

        history = controller.history
        user_messages = [m for m in history if m.role == "user"]

        assert len(user_messages) == 1
        assert user_messages[0].content == "Me explique o que é CI/CD."


# ---------------------------------------------------------------------------
# Teste 2: Validação de RECEBIMENTO
# Garante que a resposta da API é corretamente interpretada e retornada.
# ---------------------------------------------------------------------------

class TestRecebimentoResposta:
    """Testes para validar a unidade lógica de recebimento de respostas."""

    @patch("src.claude_service.anthropic.Anthropic")
    def test_retorna_conteudo_da_resposta_corretamente(self, MockAnthropic):
        """
        Verifica que o texto retornado pela API é mapeado corretamente
        para o campo `content` do ChatResponse.
        """
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        expected_text = "Brasília é a capital do Brasil."
        mock_client.messages.create.return_value = make_mock_response(text=expected_text)

        service = ClaudeService(api_key="fake-key-for-test")
        response = service.send_message("Qual é a capital do Brasil?")

        assert isinstance(response, ChatResponse)
        assert response.content == expected_text

    @patch("src.claude_service.anthropic.Anthropic")
    def test_retorna_metadados_de_uso_de_tokens(self, MockAnthropic):
        """
        Verifica que os dados de uso (tokens) são corretamente
        capturados e retornados no ChatResponse.
        """
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = make_mock_response()

        service = ClaudeService(api_key="fake-key-for-test")
        response = service.send_message("Olá!")

        assert response.input_tokens == 10
        assert response.output_tokens == 20
        assert response.model == "claude-sonnet-4-20250514"

    @patch("src.claude_service.anthropic.Anthropic")
    def test_controller_adiciona_resposta_ao_historico(self, MockAnthropic):
        """
        Verifica que o ChatController salva a resposta do assistente
        no histórico após receber da API.
        """
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = make_mock_response("Resposta do Claude.")

        service = ClaudeService(api_key="fake-key-for-test")
        controller = ChatController(service)
        controller.send("Teste.")

        assistant_messages = [m for m in controller.history if m.role == "assistant"]

        assert len(assistant_messages) == 1
        assert assistant_messages[0].content == "Resposta do Claude."
