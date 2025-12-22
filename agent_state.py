# -*- coding: utf-8 -*-
# author: Mrinaal Dogra (mrinaald)

from typing import Annotated, TypedDict, List
import operator

from langchain_core.messages import BaseMessage

# This is the "Shared Clipboard" that all agents will read from and write to.
class AgentState(TypedDict):
    # 'messages' holds the full conversation history
    # operator.add tells the graph to APPEND new messages, not overwrite them
    messages: Annotated[List[BaseMessage], operator.add]

    # 'next' tracks which agent should act next
    next: str
