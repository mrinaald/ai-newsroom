# ai-newsroom

A multi agent newsroom built with LangGraph and LangChain. The system demonstrates a practical workflow for using a local LLM with tool augmented research and a supervised handoff to a writer that produces a clean Markdown report.

## Features

- Local LLM inference using Ollama and the Llama 3.1 model
- Multi agent workflow built with LangGraph
- Researcher agent with DuckDuckGo search tool for live web information
- Writer agent that produces clean Markdown summaries
- Supervisor agent that deterministically routes work between agents and finishes the conversation
- Simple CLI runner with streaming output and recursion safeguards

## Architecture

- `agent_state.py`: Defines the shared conversation state messages and next keys
- `app.py`: Compiles and runs a LangGraph of three nodes Supervisor, Researcher, Writer
- `researcher.py`: Creates an agent with a search tool DuckDuckGoSearchRun and a focused system prompt for reliable web research
- `supervisor.py`: Provides deterministic routing and a simpler decision policy that reduces context switching errors in small models
- `writer.py`: Creates a robust writer agent with retries that turns research into a structured Markdown report

High level flow
1. Supervisor receives the user query and starts with Researcher
2. Researcher searches and returns relevant findings
3. Supervisor evaluates whether to collect more research or hand off to Writer
4. Writer produces a Markdown report
5. Supervisor detects Writer output and finishes the run

## Setup

Install Ollama

Pull Llama 3.1 model
```sh
# This pulls the 8B parameter model approximately 4.7GB.
ollama pull llama3.1
```

Setup Python virtual environment and install requirements
```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project has been tested on Python 3.12.3 on WSL2 Ubuntu.

## Usage

Activate the Python environment and run the app
```sh
source .venv/bin/activate
python app.py
```

Then enter a research topic when prompted. The terminal will stream agent messages and decisions until completion.

## Design decisions

- Deterministic routing over free form reasoning to reduce failure modes in small models
- Simple search tool with DuckDuckGo for fast and reliable web queries
- Retry and nudge loop in Writer to recover from empty outputs without polluting global state
- Clear separation of concerns per agent for maintainability and testing

## Troubleshooting

- If Ollama is not running, start it in a separate terminal
```sh
ollama serve
```

- If the model is missing, pull it
```sh
ollama pull llama3.1
```

- If you hit recursion limit, increase it with the flag
```sh
python app.py --recursion-limit 20
```

## Notes

- This project is focused on showcasing multi agent orchestration and practical tool use. It is intentionally minimal and can be extended with more agents, richer tools, and persistence.
- All outputs are streamed to the terminal for transparency and debugging.

## License

This is a personal learning project. Feel free to use and modify for educational purposes.

## Author

Mrinaal Dogra ([mrinaald](https://github.com/mrinaald))
