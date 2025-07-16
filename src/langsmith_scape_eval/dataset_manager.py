"""
Universal dataset management utilities for LangSmith evaluation.

This module provides tools for creating, updating, querying, and managing datasets
for various types of agent evaluations in LangSmith. It supports both simple Q&A
datasets and complex tool-based agent datasets with flexible data structures.
"""

import os
import json
from typing import List, Dict, Any
from datetime import datetime
from langsmith import Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class DatasetManager:
    """
    Universal dataset manager for LangSmith evaluations.
    
    This class provides a comprehensive interface for managing datasets across
    different evaluation scenarios - from simple Q&A to complex tool-based agents.
    It handles proper formatting according to LangSmith's expected structure.
    
    Attributes:
        client: LangSmith client instance for API interactions
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the dataset manager with LangSmith client.
        
        Args:
            api_key: LangSmith API key. If None, will use LANGSMITH_API_KEY env var.
        """
        self.client = Client(api_key=api_key or os.getenv("LANGSMITH_API_KEY"))
    
    def create_dataset(
        self,
        dataset_name: str,
        examples: List[Dict[str, Any]],
        description: str = "",
        dataset_type: str = "general"
    ) -> str:
        """
        Create a universal dataset in LangSmith.
        
        This method creates datasets with flexible input/output structures,
        supporting various evaluation scenarios from Q&A to tool-based agents.
        
        Args:
            dataset_name: Name of the dataset to create
            examples: List of example dictionaries with 'inputs' and 'outputs' keys
            description: Description of the dataset
            dataset_type: Type of dataset for categorization
            
        Returns:
            Dataset ID string
            
        Raises:
            LangSmithAuthError: If authentication fails
            HTTPError: If the API request fails
        """
        # Create the dataset with metadata
        full_description = f"{dataset_type.title()} dataset: {description}" if description else f"{dataset_type.title()} dataset: {dataset_name}"
        
        dataset = self.client.create_dataset(
            dataset_name=dataset_name,
            description=full_description
        )
        
        # Validate and prepare examples
        formatted_examples = []
        for i, example in enumerate(examples):
            if "inputs" not in example or "outputs" not in example:
                raise ValueError(f"Example {i} must have 'inputs' and 'outputs' keys")
            
            formatted_examples.append({
                "inputs": example["inputs"],
                "outputs": example["outputs"]
            })
        
        # Add examples to dataset
        self.client.create_examples(
            dataset_id=dataset.id,
            examples=formatted_examples
        )
        
        print(f"✅ Created {dataset_type} dataset '{dataset_name}' with {len(examples)} examples")
        return dataset.id
    
    def create_qa_dataset(
        self,
        dataset_name: str,
        qa_pairs: List[Dict[str, str]]
    ) -> str:
        """
        Create a simple Q&A dataset.
        
        Args:
            dataset_name: Name of the dataset
            qa_pairs: List of dictionaries with 'question' and 'answer' keys
            
        Returns:
            Dataset ID string
        """
        examples = []
        for pair in qa_pairs:
            example = {
                "inputs": {"question": pair["question"]},
                "outputs": {"answer": pair["answer"]}
            }
            examples.append(example)
        
        return self.create_dataset(
            dataset_name=dataset_name,
            examples=examples,
            description="Q&A evaluation dataset",
            dataset_type="qa"
        )
    
    def create_agent_dataset(
        self,
        dataset_name: str,
        queries: List[Dict[str, Any]],
        domain: str = "general"
    ) -> str:
        """
        Create a tool-based agent evaluation dataset.
        
        Args:
            dataset_name: Name of the dataset
            queries: List of query dictionaries with agent-specific structure
            domain: Domain of the dataset (shopping, news, research, etc.)
            
        Returns:
            Dataset ID string
        """
        examples = []
        for query_data in queries:
            query = query_data.pop("query")

            example = {
                "inputs": {"question": query},
                "outputs": {"answer": "-"},
                "metadata": query_data
            }
            examples.append(example)
        
        return self.create_dataset(
            dataset_name=dataset_name,
            examples=examples,
            description=f"{domain.title()} agent evaluation dataset",
            dataset_type="agent"
        )
    
    def update_dataset(
        self,
        dataset_name: str,
        new_examples: List[Dict[str, Any]]
    ) -> str:
        """
        Update an existing dataset with new examples.
        
        Args:
            dataset_name: Name of the existing dataset
            new_examples: New examples to add (with 'inputs' and 'outputs' keys)
            
        Returns:
            Dataset ID
        """
        try:
            # Get existing dataset
            dataset = self.client.read_dataset(dataset_name=dataset_name)
            
            # Add new examples to existing dataset
            self.client.create_examples(
                dataset_id=dataset.id,
                examples=new_examples
            )
            
            print(f"✅ Updated dataset '{dataset_name}' with {len(new_examples)} new examples")
            return dataset.id
            
        except Exception as e:
            print(f"❌ Failed to update dataset: {e}")
            raise
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a dataset.
        
        Args:
            dataset_name: Name of the dataset to query
            
        Returns:
            Dictionary with detailed dataset information
        """
        try:
            dataset = self.client.read_dataset(dataset_name=dataset_name)
            examples = list(self.client.list_examples(dataset_id=dataset.id))
            
            # Analyze examples to determine dataset characteristics
            input_keys = set()
            output_keys = set()
            example_types = set()
            
            for example in examples[:10]:  # Sample first 10 examples
                if hasattr(example, 'inputs') and example.inputs:
                    input_keys.update(example.inputs.keys())
                if hasattr(example, 'outputs') and example.outputs:
                    output_keys.update(example.outputs.keys())
                    
                # Determine example type
                if 'question' in str(example.inputs).lower():
                    example_types.add('qa')
                elif 'query' in str(example.inputs).lower():
                    example_types.add('agent')
            
            return {
                "id": dataset.id,
                "name": dataset.name,
                "description": dataset.description,
                "example_count": len(examples),
                "created_at": dataset.created_at,
                "modified_at": dataset.modified_at if hasattr(dataset, 'modified_at') else None,
                "input_schema": list(input_keys),
                "output_schema": list(output_keys),
                "dataset_types": list(example_types),
                "sample_examples": [
                    {
                        "inputs": example.inputs,
                        "outputs": example.outputs
                    }
                    for example in examples[:3]  # Show first 3 examples
                ]
            }
        except Exception as e:
            return {"error": f"Failed to get dataset info: {e}"}
    
    def list_datasets(self, include_details: bool = False) -> List[Dict[str, Any]]:
        """
        List all datasets in the current LangSmith project.
        
        Args:
            include_details: Whether to include detailed information for each dataset
            
        Returns:
            List of dataset information dictionaries
        """
        try:
            datasets = list(self.client.list_datasets())
            
            if not include_details:
                return [
                    {
                        "id": ds.id,
                        "name": ds.name,
                        "description": ds.description,
                        "created_at": ds.created_at
                    }
                    for ds in datasets
                ]
            else:
                # Include detailed information for each dataset
                detailed_datasets = []
                for ds in datasets:
                    try:
                        examples = list(self.client.list_examples(dataset_id=ds.id, limit=1))
                        example_count = len(list(self.client.list_examples(dataset_id=ds.id)))
                        
                        detailed_datasets.append({
                            "id": ds.id,
                            "name": ds.name,
                            "description": ds.description,
                            "created_at": ds.created_at,
                            "example_count": example_count,
                            "has_examples": example_count > 0
                        })
                    except:
                        # If we can't get details, include basic info
                        detailed_datasets.append({
                            "id": ds.id,
                            "name": ds.name,
                            "description": ds.description,
                            "created_at": ds.created_at,
                            "example_count": "unknown",
                            "has_examples": "unknown"
                        })
                
                return detailed_datasets
                
        except Exception as e:
            print(f"❌ Failed to list datasets: {e}")
            return []
    
    
    def delete_dataset(self, dataset_name: str) -> bool:
        """
        Delete a dataset from LangSmith.
        
        Args:
            dataset_name: Name of the dataset to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            dataset = self.client.read_dataset(dataset_name=dataset_name)
            self.client.delete_dataset(dataset_id=dataset.id)
            print(f"✅ Deleted dataset '{dataset_name}'")
            return True
        except Exception as e:
            print(f"❌ Failed to delete dataset '{dataset_name}': {e}")
            return False
    
    def export_dataset(self, dataset_name: str, file_path: str = None) -> str:
        """
        Export a dataset to JSON file.
        
        Args:
            dataset_name: Name of the dataset to export
            file_path: Path to save the file (optional)
            
        Returns:
            Path to the exported file
        """
        try:
            dataset_info = self.get_dataset_info(dataset_name)
            
            if "error" in dataset_info:
                raise Exception(dataset_info["error"])
            
            # Get all examples
            dataset = self.client.read_dataset(dataset_name=dataset_name)
            examples = list(self.client.list_examples(dataset_id=dataset.id))
            
            export_data = {
                "dataset_info": {
                    "name": dataset_info["name"],
                    "description": dataset_info["description"],
                    "created_at": str(dataset_info["created_at"]),
                    "example_count": dataset_info["example_count"]
                },
                "examples": [
                    {
                        "inputs": example.inputs,
                        "outputs": example.outputs
                    }
                    for example in examples
                ]
            }
            
            if not file_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"{dataset_name}_{timestamp}.json"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Exported dataset '{dataset_name}' to {file_path}")
            return file_path
            
        except Exception as e:
            print(f"❌ Failed to export dataset: {e}")
            raise
    
    def import_dataset(self, file_path: str, dataset_name: str = None) -> str:
        """
        Import a dataset from JSON file.
        
        Args:
            file_path: Path to the JSON file to import
            dataset_name: Name for the new dataset (optional, uses file name if not provided)
            
        Returns:
            Dataset ID of the imported dataset
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not dataset_name:
                # Use filename without extension as dataset name
                import os
                dataset_name = os.path.splitext(os.path.basename(file_path))[0]
            
            examples = data.get("examples", [])
            description = data.get("dataset_info", {}).get("description", f"Imported from {file_path}")
            
            dataset_id = self.create_dataset(
                dataset_name=dataset_name,
                examples=examples,
                description=description,
                dataset_type="imported"
            )
            
            print(f"✅ Imported dataset '{dataset_name}' from {file_path}")
            return dataset_id
            
        except Exception as e:
            print(f"❌ Failed to import dataset: {e}")
            raise

    # Predefined dataset creation methods
    def create_shopping_dataset(self) -> str:
        """Create a sample shopping/e-commerce dataset."""
        shopping_queries = [
            {
                "query": "남자 셔츠 추천",
                "expected_info": "남성용 셔츠 상품 정보, 브랜드, 가격, 사이즈",
                "validation_criteria": ["product_name", "brand", "price", "availability"]
            },
            {
                "query": "여자 원피스 여름",
                "expected_info": "여성용 여름 원피스 상품 정보, 스타일, 가격",
                "validation_criteria": ["product_name", "category", "price", "seasonal_info"]
            },
            {
                "query": "운동화 나이키 에어맥스",
                "expected_info": "나이키 에어맥스 운동화 상품 정보, 사이즈, 가격",
                "validation_criteria": ["product_name", "brand", "model", "price", "sizes"]
            },
            {
                "query": "겨울 패딩 추천",
                "expected_info": "겨울 패딩 상품 정보, 브랜드, 보온성, 가격",
                "validation_criteria": ["product_name", "warmth_rating", "price", "brand"]
            },
            {
                "query": "청바지 슬림핏",
                "expected_info": "슬림핏 청바지 상품 정보, 핏, 브랜드, 가격",
                "validation_criteria": ["product_name", "fit_type", "brand", "price"]
            }
        ]
        
        return self.create_agent_dataset("shopping_agent_dataset", shopping_queries, "shopping")
    
    def create_news_dataset(self) -> str:
        """Create a sample news dataset."""
        news_queries = [
            {
                "query": "최신 기술 뉴스",
                "expected_info": "최신 기술 관련 뉴스 기사 제목, 내용, 출처",
                "validation_criteria": ["article_title", "summary", "source", "date"]
            },
            {
                "query": "경제 동향 분석",
                "expected_info": "경제 동향 분석 기사, 주요 지표, 전망",
                "validation_criteria": ["economic_indicators", "analysis", "forecast"]
            },
            {
                "query": "스포츠 경기 결과",
                "expected_info": "스포츠 경기 결과, 점수, 하이라이트",
                "validation_criteria": ["match_result", "score", "highlights", "teams"]
            }
        ]
        
        return self.create_agent_dataset("news_agent_dataset", news_queries, "news")
    
    def create_research_dataset(self) -> str:
        """Create a sample research/academic dataset."""
        research_queries = [
            {
                "query": "AI 논문 최신 동향",
                "expected_info": "최신 AI 논문 제목, 저자, 초록, 발표 학회",
                "validation_criteria": ["paper_title", "authors", "abstract", "venue"]
            },
            {
                "query": "머신러닝 튜토리얼",
                "expected_info": "머신러닝 학습 자료, 튜토리얼, 코드 예제",
                "validation_criteria": ["tutorial_content", "code_examples", "difficulty_level"]
            },
            {
                "query": "데이터 사이언스 채용",
                "expected_info": "데이터 사이언스 채용 공고, 요구사항, 연봉",
                "validation_criteria": ["job_title", "requirements", "salary", "company"]
            }
        ]
        
        return self.create_agent_dataset("research_agent_dataset", research_queries, "research")


def main():
    """
    Interactive dataset management CLI.
    
    Provides a command-line interface for managing datasets with various operations.
    """
    manager = DatasetManager()
    
    print("🗃️  LangSmith Dataset Manager")
    print("=" * 40)
    
    while True:
        print("\n📋 Available Operations:")
        print("1. List all datasets")
        print("2. Get dataset details")
        print("3. Create sample datasets")
        print("4. Export dataset")
        print("5. Import dataset")
        print("6. Delete dataset")
        print("0. Exit")
        
        choice = input("\n🔢 Select operation (0-6): ").strip()
        
        try:
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                print("\n📊 All Datasets:")
                datasets = manager.list_datasets(include_details=True)
                if not datasets:
                    print("No datasets found.")
                else:
                    for ds in datasets:
                        print(f"  📁 {ds['name']} ({ds['example_count']} examples)")
                        print(f"     {ds['description']}")
                        print(f"     Created: {ds['created_at']}")
            
            elif choice == "2":
                name = input("📄 Enter dataset name: ").strip()
                info = manager.get_dataset_info(name)
                if "error" in info:
                    print(f"❌ {info['error']}")
                else:
                    print(f"\n📊 Dataset Details: {info['name']}")
                    print(f"   Description: {info['description']}")
                    print(f"   Examples: {info['example_count']}")
                    print(f"   Input Schema: {info['input_schema']}")
                    print(f"   Output Schema: {info['output_schema']}")
                    print(f"   Types: {info['dataset_types']}")
            
            elif choice == "3":
                print("\n🏗️  Creating sample datasets...")
                shopping_id = manager.create_shopping_dataset()
                news_id = manager.create_news_dataset()
                research_id = manager.create_research_dataset()
                print("✅ All sample datasets created!")
            
            elif choice == "4":
                name = input("📤 Enter dataset name to export: ").strip()
                file_path = manager.export_dataset(name)
                print(f"✅ Exported to: {file_path}")
            
            elif choice == "5":
                file_path = input("📥 Enter JSON file path to import: ").strip()
                dataset_name = input("📝 Enter new dataset name (or press Enter for auto): ").strip()
                dataset_id = manager.import_dataset(file_path, dataset_name or None)
                print(f"✅ Imported with ID: {dataset_id}")
            
            elif choice == "6":
                name = input("🗑️  Enter dataset name to delete: ").strip()
                confirm = input(f"⚠️  Are you sure you want to delete '{name}'? (y/N): ").strip().lower()
                if confirm == 'y':
                    success = manager.delete_dataset(name)
                    if success:
                        print("✅ Dataset deleted successfully!")
                else:
                    print("❌ Deletion cancelled.")
            
            else:
                print("❌ Invalid choice. Please select 0-6.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()