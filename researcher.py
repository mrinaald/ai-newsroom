# -*- coding: utf-8 -*-
# author: Mrinaal Dogra (mrinaald)

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain.agents import create_agent

from agent_state import AgentState


_NAME = "Researcher"
_SEARCH_TOOL = DuckDuckGoSearchRun()
_SYSTEM_PROMPT = "You are a web researcher. Your job is to search for accurate, reliable, and up-to-date information using the DuckDuckGo search tool. Always prioritize high-quality sources, such as trusted websites, academic papers, and reputable news outlets. Return only relevant information and avoid extraneous or conflicting details. The information should be clear and easy to understand."

def create_researcher_agent(llm: BaseChatModel):
    # researcher_llm = llm.bind_tools([_SEARCH_TOOL])
    researcher_agent = create_agent(llm, tools=[_SEARCH_TOOL])

    def agent_node(state: AgentState):
        messages = state["messages"]

        # response = researcher_llm.invoke([SystemMessage(content=_SYSTEM_PROMPT)] + messages)
        response = researcher_agent.invoke({
            "messages": [SystemMessage(content=_SYSTEM_PROMPT)] + messages
        })

        # Return the response to be appended to the state
        # We explicitly wrap the response with the sender's name
        # This helps the Supervisor 'see' who spoke last.

        # return {"messages": [AIMessage(content=response.content, name=_NAME)]}
        return {"messages": [AIMessage(content=response["messages"][-1].content, name=_NAME)]}

    return agent_node
