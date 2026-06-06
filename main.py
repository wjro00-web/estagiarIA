"""
main.py
Ponto de entrada da aplicação — interface de linha de comando (CLI).
Responsabilidade: apenas orquestrar a interação com o usuário.
"""

import os
from src.claude_service import ClaudeService
from src.chat_controller import ChatController


def run_chat() -> None:
    """Executa o loop principal do chat interativo."""
    print("=" * 55)
    print("  Claude Chat — Integração com API da Anthropic")
    print("  Digite 'sair' para encerrar | 'limpar' para reiniciar")
    print("=" * 55)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n[ERRO] Defina a variável de ambiente ANTHROPIC_API_KEY antes de executar.")
        return

    service = ClaudeService(api_key=api_key)
    controller = ChatController(service)

    while True:
        try:
            user_input = input("\nVocê: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "sair":
                print("Encerrando. Até logo!")
                break

            if user_input.lower() == "limpar":
                controller.clear_history()
                print("[Histórico limpo]")
                continue

            response = controller.send(user_input)
            print(f"\nClaude: {response.content}")
            print(f"  ↳ tokens: {response.input_tokens} entrada / {response.output_tokens} saída")

        except KeyboardInterrupt:
            print("\nEncerrando.")
            break
        except ValueError as error:
            print(f"[Entrada inválida] {error}")


if __name__ == "__main__":
    run_chat()
