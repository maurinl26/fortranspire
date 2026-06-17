"""Configuration centralisée du local_code_agent."""
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

# Charge manuellement le .env pour court-circuiter le terminal de l'IDE
load_dotenv()

@dataclass
class AgentConfig:
    # --- Modèle Mistral par défaut (La Plateforme ou endpoint compatible) ---
    model_name: str = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

    # --- Paramètres de génération ---
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    top_p: float = float(os.getenv("LLM_TOP_P", "0.9"))
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
