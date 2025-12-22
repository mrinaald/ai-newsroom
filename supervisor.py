# -*- coding: utf-8 -*-
# author: Mrinaal Dogra (mrinaald)

from langchain_core.messages import SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from agent_state import AgentState


# We force the Supervisor to pick one of these paths
_OPTIONS = ["Researcher", "Writer", "FINISH"]
_SYSTEM_PROMPT = (
    "You are a Supervisor agent managing a user query between different worker agents.\n"
    "You are assigned two different worker agents: 'Researcher' and 'Writer'.\n\n"
    "Rules:\n"
    "1. If the user query requires further information, output 'Researcher' to route the task to the Researcher agent to gather relevant and up-to-date information.\n"
    "2. If the Researcher agent has returned all necessary information, output 'Writer' to route the information to the Writer agent for summarization into a clean Markdown report.\n"
    "3. After the Writer agent outputs the final report, output 'FINISH' to end the conversation.\n\n"
    "Instructions:\n"
    "Valid Outputs are: 'Researcher', 'Writer', or 'FINISH'.\n"
    "You MUST return ONLY one of the three valid outputs ONLY.\n"
    "You MUST NOT add any extra text or context, only the name of the worker agent, or 'FINISH'.\n"
    "You MUST NOT return empty text.\n"
    "Your decisions should be based on the quality and relevance of the information provided by each agent."
)

def create_supervisor_complex_agent(supervisor_llm: BaseChatModel):
    """This node doesn't do any work per-say; it just decides."""
    def agent_node(state: AgentState):
        messages = state["messages"]

        # We invoke the LLM to make the decision
        response = supervisor_llm.invoke([SystemMessage(content=_SYSTEM_PROMPT)] + messages)

        # We clean the output to ensure it matches our options
        decision = response.content.strip()

        # Fallback: If Llama chats instead of picking, default to FINISH to prevent infinite loops
        if decision not in _OPTIONS:
            # A simple heuristic: if the last message was from Writer, we are probably done.
            if messages and "Report" in messages[-1].content:
            # if messages and messages[-1].name == "Writer":
                decision = "FINISH"
            elif "Resercher" in decision:
                decision = "Researcher"
            elif "Writer" in decision:
                decision = "Writer"
            else:
                decision = "Researcher"

        # We update the 'next' field in the state so the graph knows where to go
        return {"next": decision}

    return agent_node


_SYSTEM_PROMPT_SIMPLE = (
    "You are a Supervisor agent managing a user query between different worker agents.\n"
    "You are assigned two different worker agents: 'Researcher' and 'Writer'.\n\n"
    "Rules:\n"
    "1. If you have enough information in message history to answer user query, output 'Writer'.\n"
    "2. If you don't have enough information in message history to answer user query, output 'Researcher'.\n"
    "Instructions:\n"
    "You MUST return ONLY one of the two valid outputs: 'Researcher' or 'Writer'.\n"
    "You MUST NOT add any extra text or context, only the name of the worker agent.\n"
    "You MUST NOT return empty text.\n"
    "Your decisions should be based on the quality and relevance of the information provided by each agent."
)


def create_supervisor_agent(llm: BaseChatModel):
    """
    Uses Deterministic Routing
    8B models struggle to switch context. We help them by checking the sender.
    """
    def agent_node(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]

        # If the Writer just spoke, we are definitely done.
        if hasattr(last_message, "name") and last_message.name == "Writer":
            return {"next": "FINISH"}

        # If the Researcher just spoke, move to Writer if we have enough information.
        if hasattr(last_message, "name") and last_message.name == "Researcher":
            response = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT_SIMPLE)] + messages)
            answer = response.content.strip()
            print(f" * Supervisor response: [{answer}]")
            if "researcher" in answer.lower():
                print("Supervisor: More research needed. Handing off to Researcher.")
                return {"next": "Researcher"}
            else:
                print("Supervisor: Research data detected. Handing off to Writer.")
                return {"next": "Writer"}

        # Otherwise (Start of conversation), ask the LLM or default to Research.
        return {"next": "Researcher"}

    return agent_node
