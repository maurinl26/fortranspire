"""Point d'entrée CLI — boucle REPL interactive pour l'agent de code."""
import sys
from pathlib import Path

# Chargement optionnel du fichier .env (OLLAMA_BASE_URL, OLLAMA_MODEL, etc.)
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"[.env] Chargé depuis {env_file}")
except ImportError:
    pass

from fortranspire.agent import CodeAgent
from fortranspire.config import config


BANNER = f"""
╔══════════════════════════════════════════════════════╗
║          🤖  Local Code Agent  🤖                   ║
║  Modèle : {config.model_name:<38} ║
║  Ollama  : {config.ollama_base_url:<38} ║
╚══════════════════════════════════════════════════════╝
Tape 'exit' ou 'quit' pour quitter.
"""


def main() -> None:
    print(BANNER)

    try:
        agent = CodeAgent()
    except Exception as e:
        print(f"[Erreur] Impossible de créer l'agent : {e}")
        print("Vérifie qu'Ollama est bien démarré (docker compose up).")
        sys.exit(1)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Agent] Au revoir !")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("[Agent] Au revoir !")
            break

        try:
            response = agent.run(user_input)
            print(f"\n[Agent]\n{response}")
        except KeyboardInterrupt:
            print("\n[Agent] Requête interrompue.")
        except Exception as e:
            print(f"\n[Erreur] {e}")


if __name__ == "__main__":
    main()
