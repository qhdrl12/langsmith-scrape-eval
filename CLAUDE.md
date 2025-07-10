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

# Run the application
uv run python main.py

# Run tests (once test framework is set up)
uv run pytest

# Run linting/formatting (recommend ruff for Python projects)
uv run ruff check .
uv run ruff format .

# Add new dependencies
uv add <package_name>

# Add development dependencies
uv add --dev <package_name>
```

## Architecture Focus

This project centers around LangSmith evaluation capabilities:

1. **Dataset Management**: Creating and managing datasets for evaluation
2. **Evaluation Workflows**: Implementing evaluation pipelines using LangSmith
3. **Data Processing**: Handling data transformations and preparation for evaluation

## Key Integration Points

- LangSmith SDK for evaluation and dataset operations
- Dataset creation and management workflows
- Evaluation metric implementation and tracking
- Result analysis and reporting capabilities

## Development Notes

- Follow LangSmith best practices for evaluation setup
- Structure evaluation runs for reproducibility
- Implement proper error handling for dataset operations
- Consider async patterns for large-scale evaluations