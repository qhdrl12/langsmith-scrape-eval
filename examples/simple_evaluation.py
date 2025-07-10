"""
Simple evaluation example using LangSmith.

This script demonstrates how to run evaluations using LangSmith's evaluation framework.
It includes both a dummy model (for testing without API calls) and an OpenAI model
for real-world evaluation scenarios.

The script shows:
1. How to create model functions compatible with LangSmith
2. How to run evaluations with custom evaluators
3. How to handle results and display summaries
4. How to configure different evaluation scenarios
"""

import os
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from langchain.schema import BaseOutputParser
from dotenv import load_dotenv

# Load environment variables from .env file
# This is essential for API keys and configuration
load_dotenv()

# Add the src directory to the Python path
# This allows us to import our custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from langsmith_scape_eval.evaluators import get_default_evaluators


def simple_qa_model(inputs: dict) -> dict:
    """
    Simple Q&A model that uses OpenAI GPT for answering questions.
    
    This function serves as a model wrapper that LangSmith can call during
    evaluation. It takes the standardized input format and returns the
    standardized output format expected by LangSmith.
    
    Args:
        inputs: Dictionary containing the input data, expected to have a 'question' key
        
    Returns:
        Dictionary with 'answer' key containing the model's response
        
    Note:
        This function will make API calls to OpenAI and requires OPENAI_API_KEY
    """
    # Initialize the OpenAI LLM
    # Temperature=0 for consistent, reproducible results during evaluation
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Extract the question from the input
    question = inputs["question"]
    
    # Create a simple prompt for the model
    # The prompt instructs the model to be concise, which is good for evaluation
    prompt = f"Answer the following question concisely: {question}"
    
    # Get the model's response
    # Note: Using predict() which is deprecated but still works
    response = llm.predict(prompt)
    
    # Return in the format expected by LangSmith evaluators
    # The 'answer' key matches what our evaluators expect
    return {"answer": response}


def dummy_qa_model(inputs: dict) -> dict:
    """
    Dummy Q&A model for testing without API calls.
    
    This model provides predefined answers for a set of known questions.
    It's useful for testing the evaluation pipeline without incurring API costs
    or network dependencies. The answers are designed to match the expected
    outputs in the sample dataset.
    
    Args:
        inputs: Dictionary containing the input data with 'question' key
        
    Returns:
        Dictionary with 'answer' key containing the predefined response
        
    Note:
        This model is deterministic and doesn't require any API keys
    """
    # Extract the question from the input
    question = inputs["question"]
    
    # Define predefined responses for known questions
    # These responses are designed to match the expected answers in the sample dataset
    dummy_responses = {
        "what is the capital of france": "Paris",
        "what is 2 + 2": "4",
        "who wrote romeo and juliet": "William Shakespeare",
        "what is the largest planet in our solar system": "Jupiter",
        "what is the boiling point of water in celsius": "100 degrees Celsius"
    }
    
    # Convert question to lowercase for case-insensitive matching
    question_lower = question.lower()
    
    # Search for matching questions in our predefined responses
    for key, value in dummy_responses.items():
        if key in question_lower:
            return {"answer": value}
    
    # Default response for unknown questions
    return {"answer": "I don't know the answer to that question."}


def run_evaluation(dataset_name: str = "sample-qa-dataset", use_dummy_model: bool = True):
    """
    Run evaluation on a dataset using LangSmith's evaluation framework.
    
    This function orchestrates the entire evaluation process:
    1. Sets up the LangSmith client
    2. Configures the model to be evaluated
    3. Loads the evaluators
    4. Runs the evaluation
    5. Displays the results
    
    Args:
        dataset_name: Name of the dataset in LangSmith to evaluate against
        use_dummy_model: If True, use dummy model; if False, use OpenAI model
    """
    # Initialize the LangSmith client
    # This will use the LANGSMITH_API_KEY from environment variables
    client = Client()
    
    # Choose which model to evaluate based on the parameter
    model_function = dummy_qa_model if use_dummy_model else simple_qa_model
    model_name = "dummy-qa-model" if use_dummy_model else "openai-qa-model"
    
    print(f"Running evaluation with {model_name} on dataset: {dataset_name}")
    
    # Get the list of evaluators to use
    # This returns our custom evaluators defined in the evaluators module
    evaluators = get_default_evaluators()
    
    try:
        # Run the evaluation using LangSmith's evaluate function
        # This is the core evaluation call that:
        # 1. Loads the dataset
        # 2. Runs the model on each example
        # 3. Applies all evaluators to each result
        # 4. Aggregates the results
        results = evaluate(
            model_function,  # The model function to evaluate
            data=dataset_name,  # The dataset to evaluate against
            evaluators=evaluators,  # List of evaluators to apply
            experiment_prefix=f"eval-{model_name}",  # Prefix for experiment naming
            metadata={  # Additional metadata to store with the experiment
                "model_type": model_name,
                "dataset": dataset_name,
                "description": "Simple Q&A evaluation test"
            }
        )
        
        print(f"Evaluation completed! Results: {results}")
        
        # Print summary of results
        print("\n--- Evaluation Summary ---")
        
        # Display basic information about the evaluation
        print(f"✅ Experiment: {results.experiment_name}")
        print(f"✅ Evaluation completed successfully!")
        
        # Show available attributes for debugging purposes
        # This helps understand what data is available in the results object
        print(f"📊 Available result attributes: {[attr for attr in dir(results) if not attr.startswith('_')]}")
        
        # Try to extract and display aggregate results
        try:
            # Check different possible result formats
            # The LangSmith API may return results in different formats
            if hasattr(results, 'aggregate_metrics'):
                print("📊 Aggregate Metrics:")
                for key, value in results.aggregate_metrics.items():
                    print(f"  {key}: {value}")
            elif hasattr(results, 'results'):
                print("📊 Results available in results attribute")
            else:
                # If we can't access results locally, direct user to dashboard
                print("📋 Results stored in LangSmith (view via URL above)")
                print("🔗 Check the dashboard for detailed metrics")
                
        except Exception as e:
            # Handle any errors in result processing
            print(f"📋 Results stored in LangSmith (view via URL above)")
            print(f"🔗 Check the dashboard for detailed metrics")
        
    except Exception as e:
        # Handle evaluation errors and provide helpful debugging information
        print(f"Error running evaluation: {e}")
        print("Make sure you have:")
        print("1. LANGSMITH_API_KEY environment variable set")
        print("2. Created the dataset first (run dataset_creator.py)")
        if not use_dummy_model:
            print("3. OPENAI_API_KEY environment variable set")


def main():
    """
    Main function to run evaluation examples.
    
    This function demonstrates different evaluation scenarios:
    1. Evaluation with a dummy model (no API calls required)
    2. Evaluation with an OpenAI model (requires API key)
    
    The function includes proper error handling and user guidance for
    setting up the required environment variables.
    """
    print("LangSmith Evaluation Example")
    print("=" * 30)
    
    # Check if the required LangSmith API key is available
    if not os.getenv("LANGSMITH_API_KEY"):
        print("Warning: LANGSMITH_API_KEY not set. Please set it to run evaluations.")
        return
    
    # First evaluation: Run with dummy model
    # This doesn't require any additional API keys and is good for testing
    print("\n1. Running evaluation with dummy model...")
    run_evaluation(use_dummy_model=True)
    
    # Second evaluation: Run with OpenAI model if API key is available
    # This demonstrates real-world evaluation with an actual LLM
    if os.getenv("OPENAI_API_KEY"):
        print("\n2. Running evaluation with OpenAI model...")
        run_evaluation(use_dummy_model=False)
    else:
        print("\n2. Skipping OpenAI model evaluation (OPENAI_API_KEY not set)")


# Standard Python idiom for script execution
# This allows the file to be both imported as a module and run as a script
if __name__ == "__main__":
    main()