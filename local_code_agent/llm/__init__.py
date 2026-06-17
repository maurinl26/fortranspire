"""Interface LangChain ↔ endpoint Mistral OpenAI-compatible."""
import os
from langchain_openai import ChatOpenAI
from local_code_agent.config import config


def get_llm():
    """Retourne un client LLM vers un endpoint Mistral OpenAI-compatible.

    Cible par défaut : La Plateforme Mistral (https://api.mistral.ai/v1).
    Tout endpoint exposant l'API chat/completions au format OpenAI fonctionne
    (vLLM, TGI, Ollama, gateway interne), il suffit de positionner
    MISTRAL_ENDPOINT vers la base url voulue.

    On utilise ChatOpenAI plutôt que ChatMistralAI pour garder un client unique
    quel que soit l'hébergeur (le SDK Mistral force des chemins propres à
    api.mistral.ai qui cassent sur vLLM/TGI).
    """
    api_key  = os.getenv("MISTRAL_API_KEY")
    endpoint = os.getenv("MISTRAL_ENDPOINT", "https://api.mistral.ai/v1").rstrip("/")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY doit être défini (cf .env). "
            "MISTRAL_ENDPOINT vaut par défaut https://api.mistral.ai/v1."
        )

    target = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    return ChatOpenAI(
        base_url=endpoint,
        api_key=api_key,
        model=target,
        temperature=config.temperature,
    )


# Backward-compatibility aliases (anciennement deux fonctions identiques)
get_translator_llm = get_llm
get_reasoning_llm  = get_llm
