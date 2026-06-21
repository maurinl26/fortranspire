"""Outils de manipulation de fichiers pour l'agent."""
import os
from pathlib import Path

from langchain_core.tools import tool

"""Outils de manipulation de fichiers délégués au client IDE."""

from langchain_core.tools import tool

# Dans une architecture SaaS MCP complète où l'agent est le serveur et l'IDE le client,
# le serveur MCP N'A PAS d'outils locaux pour read_file/write_file. 
# C'est l'IDE client (Antigravity, Cline, Roo Code) qui expose ses propres outils
# de lecture/écriture de fichiers au LLM.

# Cependant, avec LangChain, pour que notre boucle ReAct puisse demander à l'IDE client
# d'exécuter l'action, nous exposons des outils "Proxy" qui retournent une instruction.
# Dans une intégration MCP asynchrone native, le serveur envoie un 'call_tool' request au client.

@tool
def read_file(path: str) -> str:
    """Demande à l'IDE du développeur de lire un fichier.

    Args:
        path: Chemin du fichier à lire.
    """
    return (
        f"ACTION REQUISE PAR LE CLIENT (IDE) : Veuillez utiliser votre outil natif "
        f"pour lire le contenu du fichier '{path}' et me le renvoyer en texte clair."
    )


@tool
def write_file(path: str, content: str) -> str:
    """Demande à l'IDE du développeur d'écrire dans un fichier.

    Args:
        path: Chemin du fichier.
        content: Contenu à écrire.
    """
    return (
        f"ACTION REQUISE PAR LE CLIENT (IDE) : Veuillez utiliser votre outil natif "
        f"pour écrire ou remplacer le contenu du fichier '{path}'. Voici le contenu :\n\n"
        f"```\n{content}\n```\n\nConfirmez une fois l'écriture effectuée."
    )


@tool
def list_directory(path: str = ".") -> str:
    """Demande à l'IDE du développeur de lister les fichiers d'un répertoire.

    Args:
        path: Chemin du répertoire.
    """
    return (
        f"ACTION REQUISE PAR LE CLIENT (IDE) : Veuillez utiliser votre outil natif "
        f"pour lister le contenu du répertoire '{path}' et me le renvoyer."
    )

