# -*- coding: utf-8 -*-
# author: Mrinaal Dogra (mrinaald)

import argparse

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from researcher import create_researcher_agent
from writer import create_writer_agent
from supervisor import create_supervisor_agent
from agent_state import AgentState


def build_newsroom_app():
    # Setup the Base LLM
    # We will use the same Llama 3.1 model for all agents, but with different system prompts.
    llm = ChatOllama(model="llama3.1", temperature=0)

    # --- Define the Workers ---

    # Researcher: Has access to search tools
    research_agent = create_researcher_agent(llm)

    # Writer: No tools, just writes
    writer_agent = create_writer_agent(llm)

    # Supervisor: Decides the flow
    supervisor_agent = create_supervisor_agent(llm)


    # Initialize the Graph
    workflow = StateGraph(AgentState)

    # Add Nodes
    # We give each function a name in the graph
    workflow.add_node("Supervisor", supervisor_agent)
    workflow.add_node("Researcher", research_agent)
    workflow.add_node("Writer", writer_agent)

    # Define Entry Point
    workflow.set_entry_point("Supervisor")

    # --- Define Edges (the Connections) ---

    # Worker -> Supervisor
    # When a worker finishes, they ALWAYS report back to the boss.
    workflow.add_edge("Researcher", "Supervisor")
    workflow.add_edge("Writer", "Supervisor")

    # Supervisor -> ? (Conditional Edge)
    # The Supervisor decides where to go next based on the 'next' field in State
    conditional_map = {
        "Researcher": "Researcher",
        "Writer": "Writer",
        "FINISH": END,
    }

    workflow.add_conditional_edges(
        "Supervisor",          # Start Node
        lambda x: x["next"],   # Function to determine path (reads 'next' from agent's state)
        conditional_map        # Map output to Target Node
    )

    # Compile the Graph
    return workflow.compile()


def _main():
    parser = argparse.ArgumentParser(description="AI Newsroom Application. Runs a multi-agent workflow to research and write reports.")
    parser.add_argument(
        "--recursion-limit", type=int, default=10,
    )
    args = parser.parse_args()

    print("The AI Newsroom is Open...")

    app = build_newsroom_app()

    user_query = input("Enter your research topic: ")

    initial_state = {"messages": [HumanMessage(content=user_query)]}

    # Run the graph
    for event in app.stream(initial_state, {"recursion_limit": args.recursion_limit}):
        for key, value in event.items():
            # print(f"\n--- Node: {key} ---")
            # If there are new messages, print the last one
            print()
            if "messages" in value:
                print("Message:")
                print(value["messages"][-1].content)

            # If the supervisor made a decision, print it
            if "next" in value:
                print(f"Decision: {value['next']}")

    print("\nProcess Completed.")


if __name__ == "__main__":
    _main()
