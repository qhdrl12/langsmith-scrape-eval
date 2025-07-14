"""
Custom evaluators for LangSmith evaluation.

This module contains various evaluators that can be used to assess the quality
of AI model responses in LangSmith evaluation runs. Each evaluator takes a run
and example as input and returns a score with reasoning.

The evaluators are designed to be used with LangSmith's evaluation framework
and are decorated with @run_evaluator to integrate properly with the system.
"""

import re
from typing import Dict, Any
from langsmith.evaluation import run_evaluator
from langchain_openai import ChatOpenAI


@run_evaluator
def exact_match_evaluator(run, example) -> Dict[str, Any]:
    """
    Evaluator that checks for exact string matches between prediction and reference.
    
    This evaluator performs a strict comparison between the model's output and
    the expected answer. It's useful for cases where the answer must be exactly
    correct (e.g., factual questions with single correct answers).
    
    Args:
        run: The execution run containing the model's output
        example: The dataset example containing the expected output
        
    Returns:
        Dict containing:
            - score: 1 if exact match, 0 otherwise
            - reason: Explanation of the evaluation result
    """
    # Extract the model's prediction from the run output
    # The 'answer' key should match the output structure defined in the model
    prediction = run.outputs.get("answer", "")
    
    # Extract the expected answer from the dataset example
    reference = example.outputs.get("answer", "")
    
    # Handle case where no reference answer is provided
    if not reference:
        return {"score": 0, "reason": "No reference provided"}
    
    # Clean both strings for comparison
    # Convert to lowercase and strip whitespace for fair comparison
    prediction_clean = prediction.strip().lower()
    reference_clean = reference.strip().lower()
    
    # Perform exact string comparison
    is_exact_match = prediction_clean == reference_clean
    
    # Return the evaluation result
    return {
        "score": 1 if is_exact_match else 0,
        "reason": "Exact match" if is_exact_match else "No exact match"
    }


@run_evaluator
def contains_answer_evaluator(run, example) -> Dict[str, Any]:
    """
    Evaluator that checks if the prediction contains the reference answer.
    
    This evaluator is more lenient than exact match - it checks if the correct
    answer is contained anywhere within the model's response. This is useful
    for cases where the model might provide additional context or explanation
    along with the correct answer.
    
    Args:
        run: The execution run containing the model's output
        example: The dataset example containing the expected output
        
    Returns:
        Dict containing:
            - score: 1 if answer is contained, 0 otherwise  
            - reason: Explanation of the evaluation result
    """
    # Extract outputs from run and example
    prediction = run.outputs.get("answer", "")
    reference = example.outputs.get("answer", "")
    
    # Validate that we have a reference to compare against
    if not reference:
        return {"score": 0, "reason": "No reference provided"}
    
    # Clean both strings for comparison
    # This normalization helps catch answers regardless of formatting
    prediction_clean = prediction.strip().lower()
    reference_clean = reference.strip().lower()
    
    # Check if the reference answer is contained in the prediction
    # This allows for partial matches and additional context
    contains_answer = reference_clean in prediction_clean
    
    return {
        "score": 1 if contains_answer else 0,
        "reason": "Contains answer" if contains_answer else "Does not contain answer"
    }


@run_evaluator
def length_evaluator(run, example, max_length: int = 200) -> Dict[str, Any]:
    """
    Evaluator that measures response length appropriateness.
    
    This evaluator checks if the model's response is within an acceptable
    length limit. It's useful for ensuring responses are concise and not
    overly verbose, which can be important for certain applications.
    
    Args:
        run: The execution run containing the model's output
        example: The dataset example (not used in this evaluator)
        max_length: Maximum allowed character length for the response
        
    Returns:
        Dict containing:
            - score: 1 if within length limit, 0 otherwise
            - reason: Explanation with actual length and limit
            - metadata: Additional data including the actual length
    """
    # Get the model's response
    prediction = run.outputs.get("answer", "")
    
    # Calculate the length of the response (after stripping whitespace)
    length = len(prediction.strip())
    
    # Check if the response is within the acceptable length
    is_appropriate_length = length <= max_length
    
    return {
        "score": 1 if is_appropriate_length else 0,
        "reason": f"Length: {length} (max: {max_length})",
        "metadata": {"length": length}  # Store actual length for analysis
    }


@run_evaluator
def numeric_accuracy_evaluator(run, example, tolerance: float = 0.01) -> Dict[str, Any]:
    """
    Evaluator for numeric answers with tolerance for floating point precision.
    
    This evaluator is specifically designed for questions that expect numeric
    answers. It extracts numbers from both the prediction and reference, then
    compares them within a specified tolerance. This is useful for mathematical
    questions, measurements, or any numeric data.
    
    Args:
        run: The execution run containing the model's output
        example: The dataset example containing the expected output
        tolerance: Acceptable difference between predicted and reference numbers
        
    Returns:
        Dict containing:
            - score: 1 if numbers match within tolerance, 0 otherwise
            - reason: Explanation of the comparison result
            - metadata: The actual numbers found (if any)
    """
    # Extract the model's prediction and reference answer
    prediction = run.outputs.get("answer", "")
    reference = example.outputs.get("answer", "")
    
    # Ensure we have a reference to compare against
    if not reference:
        return {"score": 0, "reason": "No reference provided"}
    
    try:
        # Extract numbers from both strings using regex
        # This pattern matches integers and floats, including negative numbers
        pred_numbers = re.findall(r'-?\d+\.?\d*', prediction)
        ref_numbers = re.findall(r'-?\d+\.?\d*', reference)
        
        # Check if we found numbers in both strings
        if not pred_numbers or not ref_numbers:
            return {"score": 0, "reason": "No numbers found"}
        
        # Convert the first found number in each string to float
        # We take the first number assuming it's the most relevant
        pred_num = float(pred_numbers[0])
        ref_num = float(ref_numbers[0])
        
        # Compare the numbers within the specified tolerance
        # Using absolute difference to handle both positive and negative numbers
        is_accurate = abs(pred_num - ref_num) <= tolerance
        
        return {
            "score": 1 if is_accurate else 0,
            "reason": f"Predicted: {pred_num}, Reference: {ref_num}, Tolerance: {tolerance}",
            "metadata": {"predicted": pred_num, "reference": ref_num}
        }
        
    except (ValueError, IndexError) as e:
        # Handle cases where number extraction or conversion fails
        return {"score": 0, "reason": f"Error parsing numbers: {str(e)}"}


@run_evaluator
def llm_judge_evaluator(run, example, model_name: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Evaluator that uses an LLM to judge response quality.
    
    This evaluator uses another LLM (typically GPT-4) to evaluate the quality
    of responses. It's particularly useful for subjective evaluations, complex
    reasoning tasks, or when you need more nuanced judgment than simple
    string matching can provide.
    
    Args:
        run: The execution run containing the model's output
        example: The dataset example containing the expected output and input
        model_name: The model to use for evaluation (default: gpt-4o-mini)
        
    Returns:
        Dict containing:
            - score: Numeric score from 0.0 to 1.0
            - reason: Explanation from the judging LLM
            - metadata: Full LLM response for debugging
    """
    # Extract all relevant information for the LLM judge
    prediction = run.outputs.get("answer", "")
    reference = example.outputs.get("answer", "")
    question = example.inputs.get("question", "")
    
    # Ensure we have a reference answer to compare against
    if not reference:
        return {"score": 0, "reason": "No reference provided"}
    
    try:
        # Initialize the LLM for evaluation
        # Temperature=0 for consistent, deterministic evaluations
        llm = ChatOpenAI(model=model_name, temperature=0)
        
        # Create a detailed prompt for the LLM judge
        # This prompt provides clear evaluation criteria and expected format
        prompt = f"""
        You are evaluating the quality of an AI assistant's response.

        Question: {question}
        Reference Answer: {reference}
        AI Response: {prediction}

        Please evaluate the AI response on a scale of 0-1 where:
        - 1: Perfect answer, matches reference exactly or provides equivalent information
        - 0.8: Good answer with minor issues
        - 0.6: Adequate answer but missing some information
        - 0.4: Poor answer with significant issues
        - 0.2: Very poor answer, mostly incorrect
        - 0: Completely wrong or irrelevant

        Respond with just the score (0.0-1.0) and a brief reason.
        Format: SCORE: 0.8 REASON: Good answer but slightly verbose
        """
        
        # Get the LLM's evaluation
        response = llm.predict(prompt)
        
        # Parse the LLM's response to extract score and reason
        # Use regex to find the score and reason in the expected format
        score_match = re.search(r'SCORE:\s*([0-9.]+)', response)
        reason_match = re.search(r'REASON:\s*(.+)', response)
        
        if score_match:
            # Successfully parsed the response
            score = float(score_match.group(1))
            reason = reason_match.group(1) if reason_match else "LLM evaluation"
            
            return {
                "score": score,
                "reason": reason,
                "metadata": {"llm_response": response}  # Store full response for debugging
            }
        else:
            # Failed to parse the LLM response in expected format
            return {"score": 0, "reason": "Failed to parse LLM response"}
            
    except Exception as e:
        # Handle any errors during LLM evaluation (API errors, etc.)
        return {"score": 0, "reason": f"LLM evaluation error: {str(e)}"}


def get_default_evaluators() -> list:
    """
    Get a list of default evaluators for common evaluation scenarios.
    
    This function returns a curated list of evaluators that work well together
    for general Q&A evaluation tasks. It excludes the LLM judge evaluator by
    default since it requires API calls and can be slower/more expensive.
    
    Returns:
        List of evaluator functions that can be used with LangSmith's evaluate()
        
    Note:
        The LLM judge evaluator is excluded by default to avoid additional API costs.
        Add it manually if needed: evaluators.append(llm_judge_evaluator)
    """
    return [
        exact_match_evaluator,      # Strict string matching
        contains_answer_evaluator,  # Lenient substring matching  
        length_evaluator,          # Response length validation
        numeric_accuracy_evaluator, # Numeric comparison with tolerance
        # llm_judge_evaluator,      # Commented out to avoid API costs by default
    ]