"""Outil d'exécution de commandes shell pour l'agent."""
import subprocess
from langchain_core.tools import tool


@tool
def run_shell(command: str, timeout: int = 30) -> str:
    """Exécute une commande shell et retourne stdout + stderr.

    À utiliser pour lancer des tests, des linters, git, etc.
    Les commandes destructives (rm -rf, etc.) ne sont pas bloquées
    techniquement — l'agent doit être responsable de leur usage.

    Args:
        command: La commande shell à exécuter.
        timeout: Délai maximum en secondes (défaut : 30).
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = ""
        if result.stdout:
            output += f"[stdout]\n{result.stdout}"
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        output += f"\n[returncode] {result.returncode}"
        return output.strip() or "[OK] (aucune sortie)"
    except subprocess.TimeoutExpired:
        return f"[Erreur] La commande a dépassé le timeout de {timeout}s."
    except Exception as e:
        return f"[Erreur] {e}"
