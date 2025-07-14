# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the `langsmith-scape-eval` project - a Python-based system for dataset creation, management, and evaluation using LangSmith.

## Development Environment

- **Python Version**: 3.12
- **Package Manager**: uv
- **Primary Framework**: LangSmith for evaluation workflows

## Common Commands

Since this project uses `uv` for package management:

```bash
# Install dependencies
uv sync

# Interactive dataset management (create, query, export, import)
uv run python src/langsmith_scape_eval/dataset_manager.py

# Run basic Q&A evaluation example
uv run python examples/simple_evaluation.py

# Run scraping agent evaluation (requires LangGraph agent running)
uv run python examples/scraping_evaluation.py

# Test single query before full evaluation
uv run python examples/scraping_evaluation.py test

# Run linting/formatting (recommend ruff for Python projects)
uv run ruff check .
uv run ruff format .

# Add new dependencies
uv add <package_name>

# Add development dependencies
uv add --dev <package_name>
```

## Architecture Focus

This project provides two main evaluation workflows:

### 1. Basic Q&A Evaluation
- **Purpose**: Simple question-answer model evaluation
- **Use Case**: Traditional LLM evaluation with text-based metrics
- **Files**: `dataset_manager.py`, `evaluators.py`, `simple_evaluation.py`

### 2. Tool-Based Agent Evaluation (Main Focus)
- **Purpose**: Comprehensive evaluation of LangGraph agents with tool usage
- **Use Case**: Scraping agents, search agents, any tool-calling workflows
- **Files**: `dataset_manager.py`, `scrape_langsmith_evaluators.py`, `scraping_evaluation.py`

## Key Integration Points

- **LangSmith SDK**: Dataset management and evaluation orchestration
- **LangGraph SDK**: Agent execution and streaming results
- **OpenAI API**: LLM-based evaluation (optional)
- **Tool Evaluation**: Specialized metrics for tool-calling agents

## File Structure and Responsibilities

### Dataset Management (Separate Concerns)
- `dataset_manager.py`: Universal dataset management (create, query, export, import)

### Evaluation Execution (Role Separation)
- `simple_evaluation.py`: Loads datasets + runs basic Q&A evaluation
- `scraping_evaluation.py`: Loads datasets + runs tool agent evaluation

### Evaluation Metrics
- `evaluators.py`: Basic text-based evaluators (exact match, length, numeric)
- `scrape_langsmith_evaluators.py`: Universal tool evaluators (100 pts each)

## Development Notes

- **Dataset Creation**: Always separate from evaluation execution
- **Tool Evaluation**: Use 4 universal evaluators (400 total points)
- **Agent Integration**: Requires LangGraph agent running on localhost:2024
- **Async Patterns**: Handle LangGraph streaming properly
- **Error Handling**: Comprehensive error messages with troubleshooting steps