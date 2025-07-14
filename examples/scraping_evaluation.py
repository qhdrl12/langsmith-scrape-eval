"""
LangSmith를 사용한 스크래핑 에이전트 자동 평가 예시

이 스크립트는 LangSmith 데이터셋에 등록된 쿼리들을 사용하여
스크래핑/크롤링 에이전트를 자동으로 평가하는 방법을 보여줍니다.

이 파일은 평가 실행에만 집중하며, 데이터셋 생성/수정은
dataset_manager.py에서 별도로 관리합니다.

사용 방법:
1. examples/create_scraping_datasets.py로 쇼핑 데이터셋 생성
2. 이 스크립트를 실행하여 자동 평가 수행
3. LangSmith에서 평가 결과 확인

데이터셋 형식:
- inputs: {"query": "검색 질문"}
- outputs: {"expected_info": "기대하는 정보 유형", "validation_criteria": ["검증 기준들"]}
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langsmith.evaluation import evaluate
from langgraph_sdk import get_client as get_langgraph_client
from src.langsmith_scape_eval.scrape_evaluators import get_scraping_evaluators
from src.langsmith_scape_eval.dataset_manager import DatasetManager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


async def run_scraping_agent(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    스크래핑 에이전트를 실행하고 결과를 반환하는 함수
    
    LangSmith 평가에서 사용할 실제 에이전트 실행 함수입니다.
    이 함수는 inputs를 받아서 에이전트를 실행하고 결과를 반환합니다.
    
    Args:
        inputs: {"query": "사용자 질문"}
        
    Returns:
        Dict containing:
            - final_answer: 에이전트의 최종 답변
            - messages: 원본 에이전트 메시지 리스트 (도구 호출 포함)
            - execution_time: 실행 시간 (초)
    """
    try:
        # Extract query from inputs
        query = inputs.get("query", "")
        if not query:
            return {
                "final_answer": "질문이 제공되지 않았습니다.",
                "messages": [],
                "execution_time": 0
            }
        
        # Initialize LangGraph agent client
        # Make sure your LangGraph agent is running on this URL
        base_url = "http://127.0.0.1:2024"
        client = get_langgraph_client(url=base_url)
        
        # Create a new thread for this conversation
        thread = await client.threads.create()
        start_time = time.time()
        
        # Prepare input data for the agent
        # This includes system message and user query
        input_data = {
            "messages": [
                {
                    "role": "system", 
                    "content": "너는 무신사 검색 전문가야. 사용자에 질문에 해당하는 물건을 검색해줘. 정보가 부족하더라도 최대한 잘 찾아야하며 사용자에게 추가 정보를 요구하면 안돼."
                },
                {
                    "role": "user", 
                    "content": query
                }
            ]
        }
        
        # Initialize result storage variables
        final_answer = ""
        messages = []
        
        # Execute agent and process streaming results
        async for chunk in client.runs.stream(
            thread["thread_id"],
            "agent",  # This should match your LangGraph agent name
            input=input_data,
            stream_mode="updates"
        ):
            if chunk.event == "updates":
                # Process agent messages
                if "agent" in chunk.data:
                    agent_msgs = chunk.data.get("agent", {}).get("messages", [])
                    for msg in agent_msgs:
                        messages.append(msg)
                        # Store the last message with content as final answer
                        if msg.get("content"):
                            final_answer = msg.get("content", "")
                
                # Process tool execution messages
                if "tools" in chunk.data:
                    tool_messages = chunk.data.get("tools", {}).get("messages", [])
                    for tool_msg in tool_messages:
                        messages.append(tool_msg)
        
        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Debug output (can be removed in production)
        print(f"Query: {query}")
        print(f"Final answer length: {len(final_answer)} characters")
        print(f"Total messages: {len(messages)}")
        print(f"Execution time: {execution_time:.2f} seconds")
        
        return {
            "final_answer": final_answer,
            "messages": messages,
            "execution_time": execution_time
        }
        
    except Exception as e:
        # Handle any errors during agent execution
        error_message = f"에이전트 실행 중 오류 발생: {str(e)}"
        print(f"Error: {error_message}")
        return {
            "final_answer": error_message,
            "messages": [],
            "execution_time": 0
        }


def load_dataset(dataset_name: str) -> str:
    """
    DatasetManager를 활용하여 기존 데이터셋을 로드합니다.
    
    Args:
        dataset_name: 로드할 데이터셋 이름
        
    Returns:
        str: 데이터셋의 ID
        
    Raises:
        Exception: 데이터셋이 존재하지 않는 경우
    """
    try:
        manager = DatasetManager()
        
        # Get dataset info using DatasetManager
        info = manager.get_dataset_info(dataset_name)
        
        if "error" in info:
            raise Exception(info["error"])
        
        print(f"✅ 데이터셋 로드 성공: {dataset_name} (ID: {info['id']})")
        print(f"📊 데이터셋 정보:")
        print(f"   - 이름: {info['name']}")
        print(f"   - 설명: {info['description']}")
        print(f"   - 예제 수: {info['example_count']}")
        print(f"   - 생성일: {info['created_at']}")
        
        if info.get('sample_examples'):
            print(f"📝 첫 번째 예제:")
            first_example = info['sample_examples'][0]
            print(f"   - Query: {first_example['inputs'].get('query', 'N/A')}")
            print(f"   - Expected Info: {first_example['outputs'].get('expected_info', 'N/A')}")
        
        return info['id']
        
    except Exception as e:
        error_msg = f"""
❌ 데이터셋 '{dataset_name}'을 찾을 수 없습니다.

먼저 데이터셋을 생성해주세요:

1. 간단한 데이터셋 생성:
   python examples/create_scraping_datasets.py

2. 또는 데이터셋 관리자 실행:
   python src/langsmith_scape_eval/dataset_manager.py

상세 오류: {str(e)}
        """
        print(error_msg)
        raise Exception(f"Dataset '{dataset_name}' not found. Please create it first.")


async def main():
    """
    메인 평가 실행 함수
    
    이 함수는 평가 실행에만 집중합니다:
    1. 환경 변수 및 설정 확인
    2. DatasetManager로 기존 데이터셋 로드
    3. 스크래핑 에이전트 평가 실행
    4. 결과 출력 및 요약
    """
    print("🚀 LangSmith 스크래핑 에이전트 평가 시작")
    print("=" * 50)
    
    # Step 1: Environment validation
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not langsmith_api_key:
        print("❌ 오류: LANGSMITH_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   .env 파일에 LANGSMITH_API_KEY를 설정해주세요.")
        return
    
    if not openai_api_key:
        print("⚠️  경고: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   LLM 기반 평가자가 제한될 수 있습니다.")
    
    # Step 2: Load dataset (do not create)
    dataset_name = "shopping_agent_dataset"  # Default dataset name (updated naming)
    try:
        dataset_id = load_dataset(dataset_name)
    except Exception as e:
        print(f"💥 데이터셋 로드 실패: {e}")
        return
    
    # Step 3: Get evaluators
    evaluators = get_scraping_evaluators()
    print(f"🔍 평가자 로드 완료: {len(evaluators)}개")
    
    # Step 4: Set up evaluation
    experiment_name = f"scraping_agent_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n📊 평가 설정:")
    print(f"   - 실험명: {experiment_name}")
    print(f"   - 데이터셋: {dataset_name}")
    print(f"   - 평가자 수: {len(evaluators)}")
    print(f"   - 동시 실행 수: 1 (에이전트 부하 고려)")
    
    # Step 5: Execute evaluation
    print(f"\n⏳ 평가 실행 중...")
    try:
        # Wrap async function for LangSmith evaluate
        def sync_run_scraping_agent(inputs):
            """Synchronous wrapper for the async agent function."""
            return asyncio.run(run_scraping_agent(inputs))
        
        # Run LangSmith evaluation
        results = evaluate(
            sync_run_scraping_agent,      # Function to evaluate
            data=dataset_name,            # Dataset name or ID
            evaluators=evaluators,        # List of evaluators
            experiment_prefix=experiment_name,  # Experiment name
            max_concurrency=1,            # Concurrent executions (conservative for agent load)
            description="스크래핑/크롤링 에이전트 성능 평가"
        )
        
        # Step 6: Display results
        print(f"\n🎉 평가 완료!")
        print(f"📋 실험명: {experiment_name}")
        print(f"🔗 결과 확인: https://smith.langchain.com/")
        
        # Display result summary if available
        if hasattr(results, 'results'):
            total_runs = len(results.results)
            print(f"📊 총 평가 실행 수: {total_runs}")
            
            # Calculate average scores across all evaluators
            all_scores = []
            for result in results.results:
                if hasattr(result, 'evaluation_results'):
                    total_score = sum(eval_result.score for eval_result in result.evaluation_results)
                    all_scores.append(total_score)
            
            if all_scores:
                avg_score = sum(all_scores) / len(all_scores)
                print(f"📈 평균 총점: {avg_score:.1f}/400 (4개 평가자 합계)")
        
        print(f"\n💡 팁: LangSmith 대시보드에서 상세한 평가 결과를 확인하세요!")
        
    except Exception as e:
        print(f"💥 평가 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n🔧 문제 해결 체크리스트:")
        print(f"   1. LangGraph 에이전트가 http://127.0.0.1:2024에서 실행 중인지 확인")
        print(f"   2. 에이전트 이름이 'agent'로 설정되어 있는지 확인")
        print(f"   3. 환경 변수 (LANGSMITH_API_KEY, OPENAI_API_KEY)가 올바른지 확인")
        print(f"   4. 데이터셋이 올바르게 생성되어 있는지 확인")


def run_single_evaluation():
    """
    단일 쿼리에 대한 평가 테스트
    
    전체 데이터셋 평가 전에 단일 쿼리로 테스트해볼 수 있습니다.
    이 함수는 에이전트가 제대로 작동하는지 빠르게 확인하는 용도입니다.
    """
    
    async def test_single():
        print("🧪 단일 쿼리 테스트 시작")
        print("=" * 30)
        
        # Test query
        test_input = {"query": "남자 셔츠 추천"}
        
        print(f"📝 테스트 쿼리: {test_input['query']}")
        
        # Execute agent
        result = await run_scraping_agent(test_input)
        
        # Display results
        print(f"\n📊 실행 결과:")
        print(f"   - 실행 시간: {result['execution_time']:.2f}초")
        print(f"   - 최종 답변 길이: {len(result['final_answer'])} 문자")
        print(f"   - 메시지 수: {len(result['messages'])}")
        print(f"   - 최종 답변 미리보기:")
        print(f"     {result['final_answer'][:200]}{'...' if len(result['final_answer']) > 200 else ''}")
        
        return result
    
    return asyncio.run(test_single())


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Single query test mode
        print("🧪 테스트 모드로 실행 중...")
        run_single_evaluation()
    else:
        # Full evaluation mode
        print("🚀 전체 평가 모드로 실행 중...")
        asyncio.run(main())