# -*- coding: utf-8 -*-
# author: Mrinaal Dogra (mrinaald)

import time

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from agent_state import AgentState


_NAME = "Writer"
_SYSTEM_PROMPT = "You are a senior technical writer. Your task is to summarize the information provided by the Researcher agent into a clean, well-organized Markdown report. Use headings, bullet points, and clear sections to ensure the report is easy to read and understand. Ensure all information is factual and relevant, and avoid unnecessary details. Do not use emojis or casual language."
_MAX_RETRIES = 3

def create_writer_agent(writer_llm: BaseChatModel):
    def agent_node(state: AgentState):
        messages = state["messages"]
        current_messages = [SystemMessage(content=_SYSTEM_PROMPT)] + messages

        for attempt in range(_MAX_RETRIES):
            try:
                response = writer_llm.invoke(current_messages)

                # Check if content is valid
                if response.content and response.content.strip():
                    # Success
                    # Return the response to be appended to the state
                    # We explicitly wrap the response with the sender's name
                    # This helps the Supervisor 'see' who spoke last.
                    return {"messages": [AIMessage(content=response.content, name=_NAME)]}

                # If we get here, the content was empty.
                print(f"{_NAME}: returned empty response. Retrying ({attempt + 1}/{_MAX_RETRIES})...")

                # If we failed, we append a fake user message to the NEXT attempt
                # telling the model to fix itself.
                # NOTE: This is a great techinque to recover from LLM failures without modifying global state.
                print(f"{_NAME}: returned empty. Nudging...")

                # Add a nudge to the CONTEXT for the next loop (without modifying global state)
                nudge_msg = HumanMessage(content="Your last response was empty. Please try again. Ignore any complex formatting and just output the text.")
                current_messages.append(nudge_msg)

            except Exception as e:
                print(f"Error invoking {_NAME}: {e}")

            # Optional: Short sleep to let system/VRAM settle
            time.sleep(1)

        # Fallback if all retries fail
        print(f"{_NAME} failed to generate content after {_MAX_RETRIES} attempts.")
        return {"messages": [AIMessage(content="Error: Could not generate report.", name=_NAME)]}

    return agent_node
