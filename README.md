# Supervisor Agent

LangGraph-based supervisor agent with depth-aware planning and execution.
It classifies query depth, creates a task plan when needed, runs tasks (serial or parallel),
and aggregates results into a final answer.

## Features
- Depth classification (d0-d5) with a fast "direct answer" path for trivial queries.
- Planner that generates structured tasks with an execution mode.
- Executor with serial or parallel task processing.
- Search-augmented self-ask loop (Google Serper) for task execution.
- Optional LangSmith tracing support.

## Architecture (high level)
1. Analyze depth
2. Plan (if depth >= d1)
3. Execute tasks (serial or parallel)
4. Aggregate results

## Requirements
- Python 3.10+
- OpenAI API key
- Google Serper API key (for web search)

## Setup
Option A: uv
```bash
uv sync
```

Option B: pip
```bash
python -m venv .venv
. .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -e .
```

## Environment variables
Create a `.env` file at the repo root:
```
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5-mini
SERPER_API_KEY=...
LANGSMITH_API_KEY=...        # optional
LANGSMITH_PROJECT=supervisor # optional
LANGSMITH_TRACING_V2=true    # optional
EXECUTION_MODE=serial        # default is serial
```

Notes:
- `SERPER_API_KEY` is required by `GoogleSerperAPIWrapper`.
- `OPENAI_MODEL` defaults to `gpt-5-mini`.

## Usage
Run once from the CLI:
```bash
supervisor-agent "Summarize the latest developments in edge AI."
```

Choose execution mode:
```bash
supervisor-agent "Plan a weekend trip to Busan." --mode parallel
```

## Project layout
```
src/supervisor_agent/
  analyzer.py      # depth classification
  planner.py       # task plan creation
  executor.py      # task execution
  search_agent.py  # self-ask + search
  aggregator.py    # final response synthesis
  graph.py         # LangGraph wiring
  main.py          # CLI entrypoint
```
