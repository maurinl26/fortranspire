"""Agent de code basé sur LangChain ReAct + Ollama (Mistral NeMo 12B)."""
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate

from fortranspire.llm import get_llm
from fortranspire.tools import ALL_TOOLS
from fortranspire.prompts import SYSTEM_PROMPT
from fortranspire.config import config


_REACT_TEMPLATE = (
    SYSTEM_PROMPT
    + """

## Outils disponibles
{tools}

## Noms des outils (à utiliser exactement)
{tool_names}

## Historique de conversation
{chat_history}

## Question de l'utilisateur
{input}

## Plan d'action
{agent_scratchpad}"""
)


def build_agent_executor() -> AgentExecutor:
    """Construit et retourne un AgentExecutor LangChain."""
    llm = get_llm()

    prompt = PromptTemplate(
        input_variables=["tools", "tool_names", "chat_history", "input", "agent_scratchpad"],
        template=_REACT_TEMPLATE,
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=config.memory_window,
        return_messages=False,
    )

    agent = create_react_agent(llm=llm, tools=ALL_TOOLS, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        memory=memory,
        max_iterations=config.max_iterations,
        handle_parsing_errors=True,
        verbose=True,
    )


class CodeAgent:
    """Façade haut niveau pour l'agent de code."""

    def __init__(self):
        self.executor = build_agent_executor()

    def run(self, user_input: str) -> str:
        """Envoie une requête à l'agent et retourne la réponse finale."""
        result = self.executor.invoke({"input": user_input})
        return result.get("output", "")
