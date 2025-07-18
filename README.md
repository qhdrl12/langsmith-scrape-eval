# 🚀 LangSmith Scape Eval

A comprehensive Python framework for dataset creation, management, and evaluation using LangSmith.

## ✨ Features

- 📊 **Dataset Creation**: Create and manage Q&A datasets in LangSmith
- 🔍 **Custom Evaluators**: Multiple evaluation metrics for comprehensive assessment
- 🤖 **Model Testing**: Support for both dummy models and real LLM models
- 📈 **Evaluation Pipeline**: Complete workflow from dataset creation to result analysis
- 🔒 **Secure Configuration**: Environment-based API key management

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/qhdrl12/langsmith-scrape-eval.git
   cd langsmith-scrape-eval
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```env
   LANGSMITH_API_KEY=your_langsmith_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here  # Optional
   ```

### Required API Keys

| Service | Required | Purpose | Get API Key |
|---------|----------|---------|-------------|
| LangSmith | ✅ Yes | Dataset management & evaluation | [smith.langchain.com](https://smith.langchain.com) |
| OpenAI | ⚪ Optional | LLM-based evaluations | [platform.openai.com](https://platform.openai.com) |

## 🚀 Quick Start

### 1. Create a Sample Dataset
```bash
uv run python src/langsmith_scape_eval/dataset_creator.py
```

### 2. Run Evaluation
```bash
uv run python examples/simple_evaluation.py
```

This will run evaluations with both a dummy model (no API calls) and OpenAI model (if API key is provided).

## 📁 Project Structure

```
langsmith-scape-eval/
├── src/langsmith_scape_eval/
│   ├── __init__.py
│   ├── dataset_creator.py              # Dataset creation utilities  
│   ├── evaluators.py                   # Basic Q&A evaluation metrics
│   ├── scrape_langsmith_evaluators.py  # Universal tool evaluators (main)
│   └── scrape_evaluator.py             # Standalone evaluation runner
├── examples/
│   ├── simple_evaluation.py            # Basic Q&A evaluation workflow
│   └── langsmith_scraping_evaluation.py # Tool-based agent evaluation
├── .env.example                        # Environment template
├── pyproject.toml                     # Project dependencies 
└── README.md                          # This file
```

## 🔍 Available Evaluators

### Universal Tool Evaluators (Main)
| Evaluator | Score | Description | Use Case |
|-----------|-------|-------------|----------|
| **Tool Selection** | 100 pts | Evaluates tool choice appropriateness | LangGraph agents with tool calling |
| **Tool Execution** | 100 pts | Assesses tool execution quality and success rate | Tool-based workflows |
| **Data Extraction** | 100 pts | Measures data collection effectiveness | Scraping, API, search agents |
| **Answer Quality** | 100 pts | Evaluates final response quality | Any agent producing final answers |

### Basic Q&A Evaluators
| Evaluator | Description | Use Case |
|-----------|-------------|----------|
| **Exact Match** | Strict string comparison | Factual questions with single correct answers |
| **Contains Answer** | Checks if answer is contained in response | Responses with additional context |
| **Length Check** | Validates response length | Ensuring concise responses |
| **Numeric Accuracy** | Compares numeric values with tolerance | Mathematical or measurement questions |
| **LLM Judge** | Uses another LLM to evaluate quality | Subjective or complex reasoning tasks |

## 💻 Usage Examples

### Creating Custom Datasets

```python
from langsmith_scape_eval.dataset_creator import DatasetCreator

# Initialize creator
creator = DatasetCreator()

# Create custom Q&A dataset
qa_pairs = [
    {"question": "What is Python?", "answer": "A programming language"},
    {"question": "What is 5 + 3?", "answer": "8"}
]

dataset_id = creator.create_qa_dataset("my-custom-dataset", qa_pairs)
```

### Running Tool-Based Agent Evaluations

```python
from langsmith.evaluation import evaluate
from langsmith_scape_eval.scrape_langsmith_evaluators import get_tool_evaluators

def my_langgraph_agent(inputs):
    # Your LangGraph agent implementation
    query = inputs["query"]
    # Execute agent and return structured output
    return {
        "final_answer": "Agent's final response",
        "agent_messages": [/* LangGraph message array */],
        "execution_time": 2.5
    }

# Run evaluation with universal tool evaluators
evaluators = get_tool_evaluators()  # 4 evaluators, 400 total points
results = evaluate(
    my_langgraph_agent,
    data="agent-evaluation-dataset", 
    evaluators=evaluators,
    experiment_prefix="tool-agent-eval"
)
```

### Running Basic Q&A Evaluations

```python
from langsmith.evaluation import evaluate
from langsmith_scape_eval.evaluators import get_default_evaluators

def my_qa_model(inputs):
    # Your Q&A model implementation
    question = inputs["question"]
    # Process question and return answer
    return {"answer": "Your model's response"}

# Run evaluation with basic evaluators
evaluators = get_default_evaluators()
results = evaluate(
    my_qa_model,
    data="qa-dataset",
    evaluators=evaluators,
    experiment_prefix="qa-experiment"
)
```

### Using Individual Evaluators

```python
# Tool evaluators for specific aspects
from langsmith_scape_eval.scrape_langsmith_evaluators import (
    tool_selection_evaluator,
    data_extraction_evaluator
)

# Basic Q&A evaluators  
from langsmith_scape_eval.evaluators import (
    exact_match_evaluator,
    numeric_accuracy_evaluator
)

# Use specific evaluators for targeted evaluation
selected_evaluators = [
    tool_selection_evaluator,  # Only evaluate tool selection (100 pts)
    data_extraction_evaluator  # Only evaluate data extraction (100 pts)
]

results = evaluate(
    my_agent,
    data="agent-dataset",
    evaluators=selected_evaluators
)
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `LANGSMITH_API_KEY` | LangSmith API authentication | ✅ |
| `LANGSMITH_ENDPOINT` | LangSmith API endpoint | ⚪ |
| `LANGSMITH_PROJECT` | Default project name | ⚪ |
| `OPENAI_API_KEY` | OpenAI API for LLM evaluations | ⚪ |

### Evaluator Parameters

You can customize evaluator behavior:

```python
# Length evaluator with custom max length
from langsmith_scape_eval.evaluators import length_evaluator

# Numeric evaluator with custom tolerance
from langsmith_scape_eval.evaluators import numeric_accuracy_evaluator

# Use in evaluation with custom parameters
# Note: Parameters are set when defining the evaluator function
```

## 📊 Viewing Results

### LangSmith Dashboard
- Evaluation results are automatically stored in LangSmith
- View detailed metrics and comparisons in the web dashboard
- Access via the URLs provided in the console output

### Console Output
- Basic experiment information
- Links to detailed results
- Error messages and debugging information

## 🛡️ Security Best Practices

- ✅ Never commit `.env` files to version control
- ✅ Use `.env.example` as a template for required variables
- ✅ Rotate API keys regularly
- ✅ Use minimal required permissions for API keys

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

**Q: "Authentication failed" error**
```
A: Check that LANGSMITH_API_KEY is correctly set in your .env file
```

**Q: "Dataset not found" error**
```
A: Make sure to create the dataset first using dataset_creator.py
```

**Q: OpenAI evaluations not working**
```
A: Verify OPENAI_API_KEY is set, or use dummy model for testing
```

### Getting Help

- 📖 Check the [LangSmith Documentation](https://docs.smith.langchain.com/)
- 🐛 Report issues on [GitHub Issues](https://github.com/qhdrl12/langsmith-scrape-eval/issues)
- 💬 Ask questions in the discussions tab

---

**Built with ❤️ using LangSmith and LangChain**