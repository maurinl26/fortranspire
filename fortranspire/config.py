"""Configuration centralisée du fortranspire."""
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

# Charge manuellement le .env pour court-circuiter le terminal de l'IDE
load_dotenv()

@dataclass
class AgentConfig:
    # --- Modèle Mistral par défaut (La Plateforme ou endpoint compatible) ---
    # NOTE (reproducibility): `-latest` is a MOVING tag — Mistral repoints it over
    # time, so a run is not reproducible across dates. Pin a dated version
    # (e.g. MISTRAL_MODEL=mistral-large-2411) for a reproducible port.
    model_name: str = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

    # --- Génération ---
    # temperature defaults to 0.0: a code *translation* must be reproducible —
    # the same Fortran + the same model must yield the same JAX/OpenACC. A
    # non-zero default made two runs of the same port diverge. Raise it
    # deliberately (LLM_TEMPERATURE) only if you want sampling.
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    top_p: float = float(os.getenv("LLM_TOP_P", "1.0"))
    num_predict: int = int(os.getenv("LLM_NUM_PREDICT", "2048"))

    # --- Agent ---
    max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "15"))
    memory_window: int = int(os.getenv("AGENT_MEMORY_WINDOW", "10"))

    # --- Répertoire de travail par défaut ---
    workspace_dir: str = os.getenv(
        "AGENT_WORKSPACE",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


# Instance partagée
config = AgentConfig()
