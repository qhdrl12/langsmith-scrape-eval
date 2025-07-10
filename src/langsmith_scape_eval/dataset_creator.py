"""
Dataset creation utilities for LangSmith evaluation.

This module provides tools for creating and managing datasets in LangSmith,
which is essential for evaluation workflows. It includes utilities for creating
Q&A datasets with proper formatting and structure expected by LangSmith.
"""

import os
from typing import List, Dict, Any
from langsmith import Client
from dotenv import load_dotenv

# Load environment variables from .env file
# This is crucial for loading API keys and other configuration
load_dotenv()


class DatasetCreator:
    """
    Create and manage datasets in LangSmith for evaluation purposes.
    
    This class provides a high-level interface for creating datasets that can be
    used in LangSmith evaluations. It handles the proper formatting of data
    according to LangSmith's expected structure.
    
    Attributes:
        client: LangSmith client instance for API interactions
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the dataset creator with LangSmith client.
        
        The client will be used to interact with LangSmith's API for creating
        datasets and adding examples. If no API key is provided, it will
        attempt to use the LANGSMITH_API_KEY environment variable.
        
        Args:
            api_key: LangSmith API key. If None, will use LANGSMITH_API_KEY env var.
        """
        # Initialize the LangSmith client
        # The Client class handles authentication and API communication
        self.client = Client(api_key=api_key)
    
    def create_qa_dataset(self, dataset_name: str, qa_pairs: List[Dict[str, str]]) -> str:
        """
        Create a Q&A dataset in LangSmith with the specified question-answer pairs.
        
        This method creates a new dataset in LangSmith and populates it with
        the provided Q&A pairs. The dataset follows LangSmith's expected format
        with 'inputs' containing questions and 'outputs' containing answers.
        
        Args:
            dataset_name: Name of the dataset to create in LangSmith
            qa_pairs: List of dictionaries, each containing 'question' and 'answer' keys
            
        Returns:
            Dataset ID string that can be used to reference the dataset in evaluations
            
        Raises:
            LangSmithAuthError: If authentication fails
            HTTPError: If the API request fails
        """
        # Step 1: Create the dataset in LangSmith
        # This creates an empty dataset container with metadata
        dataset = self.client.create_dataset(
            dataset_name=dataset_name,
            description=f"Q&A dataset: {dataset_name}"
        )
        
        # Step 2: Prepare examples in the format expected by LangSmith
        # Each example must have 'inputs' and 'outputs' dictionaries
        examples = []
        for i, pair in enumerate(qa_pairs):
            # Transform the simple dict format into LangSmith's expected structure
            example = {
                "inputs": {"question": pair["question"]},    # Input to the model
                "outputs": {"answer": pair["answer"]}        # Expected output (ground truth)
            }
            examples.append(example)
        
        # Step 3: Add all examples to the dataset in a batch operation
        # This is more efficient than adding examples one by one
        self.client.create_examples(
            dataset_id=dataset.id,
            examples=examples
        )
        
        # Provide feedback to the user about the dataset creation
        print(f"Created dataset '{dataset_name}' with {len(qa_pairs)} examples")
        return dataset.id
    
    def create_sample_qa_dataset(self) -> str:
        """
        Create a sample Q&A dataset for testing and demonstration purposes.
        
        This method creates a predefined dataset with 5 sample Q&A pairs covering
        different types of questions (factual, mathematical, literary, scientific).
        This is useful for testing evaluation workflows without having to prepare
        custom data.
        
        Returns:
            Dataset ID string for the created sample dataset
        """
        # Define sample data covering various question types
        # This provides a good mix for testing different evaluation scenarios
        sample_data = [
            {
                "question": "What is the capital of France?",
                "answer": "Paris"
            },
            {
                "question": "What is 2 + 2?",
                "answer": "4"
            },
            {
                "question": "Who wrote Romeo and Juliet?",
                "answer": "William Shakespeare"
            },
            {
                "question": "What is the largest planet in our solar system?",
                "answer": "Jupiter"
            },
            {
                "question": "What is the boiling point of water in Celsius?",
                "answer": "100 degrees Celsius"
            }
        ]
        
        # Create the dataset using the general create_qa_dataset method
        # This demonstrates reusability of the core dataset creation logic
        return self.create_qa_dataset("sample-qa-dataset", sample_data)


def main():
    """
    Main function to demonstrate dataset creation.
    
    This function serves as an example of how to use the DatasetCreator class
    to create a sample dataset. It can be run directly to quickly set up
    a test dataset for evaluation workflows.
    """
    # Create an instance of the dataset creator
    # This will use the API key from environment variables
    creator = DatasetCreator()
    
    # Create the sample dataset and get its ID
    dataset_id = creator.create_sample_qa_dataset()
    
    # Display the dataset ID for reference
    # This ID can be used in evaluation scripts to reference the dataset
    print(f"Sample dataset created with ID: {dataset_id}")


# Standard Python idiom for script execution
# This allows the file to be both imported as a module and run as a script
if __name__ == "__main__":
    main()