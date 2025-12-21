from typing import Annotated, List, TypedDict, Union, Literal
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import operator

# 1. Define the State
# This is the "Shared Clipboard" that all agents will read from and write to.
class AgentState(TypedDict):
    # 'messages' holds the full conversation history
    # operator.add tells the graph to APPEND new messages, not overwrite them
    messages: Annotated[List[BaseMessage], operator.add]
    # 'next' tracks which agent should act next
    next: str

# 2. Setup the Tools
search_tool = DuckDuckGoSearchRun()

# 3. Setup the Base LLM
# We will use the same Llama 3.1 model for all agents, but with different system prompts.
llm = ChatOllama(model="llama3.1", temperature=0)

# Helper to create an agent node
# def create_agent(llm, tools, system_prompt):
#     # If tools are provided, bind them
#     if tools:
#         llm = llm.bind_tools(tools)

#     prompt = f"System: {system_prompt}\n"

#     def agent_node(state: AgentState):
#         # The agent sees the full history
#         messages = state["messages"]
#         # We prepend the system prompt instructions
#         # Note: In a real app, you'd insert this as a proper SystemMessage
#         # But for simple Ollama/LangGraph chains, this works well.
#         response = llm.invoke([HumanMessage(content=prompt)] + messages)
#         # Return the response to be appended to the state
#         return {"messages": [response]}

#     return agent_node

def create_agent(llm, tools, system_prompt, name):
    if tools:
        llm = llm.bind_tools(tools)

    # prompt = f"System: {system_prompt}\n"

    def agent_node(state: AgentState):
        messages = state["messages"]
        print()
        print(f"--- {name} ---")
        print(f"Num messages: {len(messages)}")
        print()

        # response = llm.invoke([HumanMessage(content=prompt)] + messages)
        response = llm.invoke([SystemMessage(content=system_prompt)] + messages)

        # FIX: We explicitly wrap the response with the sender's name
        # This helps the Supervisor 'see' who spoke last.
        return {"messages": [AIMessage(content=response.content, name=name)]}

    return agent_node

# --- Define the Workers ---

# 1. Researcher: Has access to search tools
research_agent = create_agent(
    llm,
    [search_tool],
    system_prompt="You are a web researcher. You search for accurate information using the DuckDuckGo tool.",
    name="Researcher",
)

# 2. Writer: No tools, just writes
writer_agent = create_agent(
    llm,
    [],
    system_prompt="You are a senior technical writer. You summarize the conversation into a clean Markdown report. No emojis.",
    name="Summarizer",
)


# 1. The Supervisor Node
# This node doesn't "work"; it just decides.
def supervisor_node(state: AgentState):
    messages = state["messages"]

    # We force the Supervisor to pick one of these paths
    options = ["Researcher", "Summarizer", "FINISH"]

    # system_prompt = (
    #     "You are a Supervisor managing a conversation between workers.\n"
    #     "1. If detailed information is missing, output 'Researcher'.\n"
    #     "2. If you have enough information and need a final report, output 'Writer'.\n"
    #     "3. If the report is already written and satisfactory, output 'FINISH'.\n"
    #     "Return ONLY the worker name or 'FINISH'. Do not add extra text."
    # )
    system_prompt = (
        "You are a Supervisor agent managing a user query between different worker agents.\n"
        "You are assigned two different worker agents: 'Researcher' and 'Summarizer'.\n\n"
        "Rules:\n"
        "1. First, route the user query to the Researcher agent to gather relevant and up-to-date information for the user query. Output 'Researcher', which will route the task to 'Researcher' agent.\n"
        "2. Once 'Researcher' responds back, output 'Summarizer', which will route the information to 'Summarizer' agent to generate final report.\n"
        "3. Finally after 'Summarizer' responds back, output 'FINISH' to finish the response to user query.\n\n\n"
        "Instructions:\n"
        "Valid Outputs are: 'Researcher', 'Summarizer', or 'FINISH'.\n"
        "You MUST return ONLY one of the three valid outputs ONLY.\n"
        "You MUST NOT add any extra text, only the name of the worker agent, or 'FINISH'.\n"
        "You MUST NOT return empty text."
    )

    # We invoke the LLM to make the decision
    response = llm.invoke([SystemMessage(content=system_prompt)] + messages)

    # We clean the output to ensure it matches our options
    decision = response.content
    print()
    print("--- Supervisor Decision Raw Output ---")
    print(f"Num messages: {len(messages)}")
    # for msg in messages:
    #     print(f"{msg.name if msg.name else 'User'}: {msg.content}")
    #     print("++++++++++++++++++")
    print(f"Decision: [{decision}]")
    print()

    decision = decision.strip()

    # if not decision:
    #     decision = "Researcher"

    # Fallback: If Llama chats instead of picking, default to FINISH to prevent infinite loops
    if decision not in options:
        # # A simple heuristic: if the last message was from Writer, we are probably done.
        # # if messages and "Report" in messages[-1].content:
        # if messages and messages[-1].name == "Writer":
        #     decision = "FINISH"
        # else:
        #     decision = "Researcher"
        if "Resercher" in decision:
            decision = "Researcher"
        elif "Summarizer" in decision:
            decision = "Summarizer"
        else:
            decision = ""

    # We update the 'next' field in the state so the graph knows where to go
    return {"next": decision}



# 1. Initialize the Graph
workflow = StateGraph(AgentState)

# 2. Add Nodes
# We give each function a name in the graph
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Researcher", research_agent)
workflow.add_node("Summarizer", writer_agent)

# 3. Define Entry Point
# The graph always starts with the Supervisor
workflow.set_entry_point("Supervisor")

# 4. Define Edges (The Connections)

# Worker -> Supervisor
# When a worker finishes, they ALWAYS report back to the boss.
workflow.add_edge("Researcher", "Supervisor")
workflow.add_edge("Summarizer", "Supervisor")

# Supervisor -> ? (Conditional Edge)
# The Supervisor decides where to go next based on the 'next' field in State
conditional_map = {
    "": "Supervisor",  # Default to self-loop if empty
    "Researcher": "Researcher",
    "Summarizer": "Summarizer",
    "FINISH": END,
    # "Finish": END,
    # "End": END,
}

workflow.add_conditional_edges(
    "Supervisor",          # Start Node
    lambda x: x["next"],   # Function to determine path (reads 'next' from state)
    conditional_map        # Map output to Target Node
)

# 5. Compile the Graph
app = workflow.compile()



if __name__ == "__main__":
    print("📰 The AI Newsroom is Open...")

    # The initial request
    user_query = "Research the latest features of Python 3.13 and write a short summary report."

    initial_state = {"messages": [HumanMessage(content=user_query)]}

    # Run the graph
    # recursion_limit=10 prevents infinite loops if the Supervisor gets confused
    for event in app.stream(initial_state, {"recursion_limit": 10}):
        for key, value in event.items():
            print(f"\n--- 🟢 Node: {key} ---")
            # If there are new messages, print the last one
            if "messages" in value:
                print(value["messages"][-1].content)
                # print("-------------------------")
                # print("=== History ===")
                # for msg in value["messages"]:
                #     print(f"{msg.content}")
                #     print("==================")
                # print("-------------------------")
            # If the supervisor made a decision, print it
            if "next" in value:
                print(f"👉 Decision: {value['next']}")

    print("\n✅ Process Completed.")