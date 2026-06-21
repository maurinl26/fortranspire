"""Outil de recherche dans le codebase."""
import subprocess
from pathlib import Path
from langchain_core.tools import tool
from fortranspire.config import config


@tool
def search_code(pattern: str, directory: str = ".", file_glob: str = "*") -> str:
    """Recherche un pattern (regex) dans les fichiers du workspace.

    Args:
        pattern: Expression de recherche (compatible grep -E).
        directory: Répertoire dans lequel chercher (relatif au workspace).
        file_glob: Glob pour filtrer les fichiers (ex: '*.py', '*.f90').
    """
    base = Path(config.workspace_dir)
    target = (base / directory).resolve()
    if not target.exists():
        return f"[Erreur] Répertoire introuvable : {target}"

    cmd = ["grep", "-rn", "--include", file_glob, "-E", pattern, str(target)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 1:
            return "[Résultat] Aucune occurrence trouvée."
        if result.returncode != 0:
            return f"[Erreur grep] {result.stderr.strip()}"
        lines = result.stdout.strip().splitlines()
        # Limiter à 50 résultats pour éviter de surcharger le contexte
        if len(lines) > 50:
            truncated = len(lines) - 50
            lines = lines[:50] + [f"... ({truncated} lignes supplémentaires tronquées)"]
        return "\n".join(lines)
    except subprocess.TimeoutExpired:
        return "[Erreur] Recherche trop longue (timeout 15s)."
    except Exception as e:
        return f"[Erreur] {e}"
