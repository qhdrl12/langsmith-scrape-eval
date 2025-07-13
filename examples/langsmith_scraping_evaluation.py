"""
LangSmith를 사용한 스크래핑 에이전트 자동 평가 예시

이 스크립트는 LangSmith 데이터셋에 등록된 쿼리들을 사용하여
스크래핑/크롤링 에이전트를 자동으로 평가하는 방법을 보여줍니다.

사용 방법:
1. LangSmith에 평가용 데이터셋 생성
2. 이 스크립트를 실행하여 자동 평가 수행
3. LangSmith에서 평가 결과 확인

데이터셋 형식:
- inputs: {"query": "검색 질문"}
- outputs: {"expected_info": "기대하는 정보 유형"} (선택사항)
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langsmith import Client
from langsmith.evaluation import evaluate
from langgraph_sdk import get_client as get_langgraph_client
from src.langsmith_scape_eval.scrape_langsmith_evaluators import get_scraping_evaluators
from dotenv import load_dotenv

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
            - execution_time: 실행 시간
    """
    try:
        query = inputs.get("query", "")
        if not query:
            return {
                "final_answer": "질문이 제공되지 않았습니다.",
                "messages": [],
                "execution_time": 0
            }
        
        # LangGraph 에이전트 클라이언트 초기화
        base_url = "http://127.0.0.1:2024"
        client = get_langgraph_client(url=base_url)
        
        # 새로운 스레드 생성
        thread = await client.threads.create()
        start_time = time.time()
        
        # 에이전트 실행용 메시지 구성
        input_data = {
            "messages": [
                {"role": "system", "content": "너는 무신사 검색 전문가야. 사용자에 질문에 해당하는 물건을 검색해줘. 정보가 부족하더라도 최대한 잘 찾아야하며 사용자에게 추가 정보를 요구하면 안돼."},
                {"role": "user", "content": query}
            ]
        }
        
        # 결과 저장용 변수
        final_answer = ""
        messages = []
        
        # 에이전트 실행 및 스트림 처리
        async for chunk in client.runs.stream(
            thread["thread_id"],
            "agent",
            input=input_data,
            stream_mode="updates"
        ):
            if chunk.event == "updates":
                # Agent 메시지 처리
                if "agent" in chunk.data:
                    agent_msgs = chunk.data.get("agent", {}).get("messages", [])
                    for msg in agent_msgs:
                        messages.append(msg)
                        # 내용이 있는 마지막 agent 메시지를 final_answer로 저장
                        if msg.get("content"):
                            final_answer = msg.get("content", "")
                
                # 도구 실행 결과 메시지 처리
                if "tools" in chunk.data:
                    tool_messages = chunk.data.get("tools", {}).get("messages", [])
                    for tool_msg in tool_messages:
                        messages.append(tool_msg)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"final_answer: {final_answer}")
        print(f"messages: {messages}")

        return {
            "final_answer": final_answer,
            "messages": messages,
            "execution_time": execution_time
        }
        
    except Exception as e:
        return {
            "final_answer": f"에이전트 실행 중 오류 발생: {str(e)}",
            "messages": [],
            "execution_time": 0
        }


def create_sample_dataset(client: Client, dataset_name: str) -> str:
    """
    평가용 샘플 데이터셋을 생성합니다.
    
    Args:
        client: LangSmith 클라이언트
        dataset_name: 생성할 데이터셋 이름
        
    Returns:
        str: 생성된 데이터셋의 ID
    """
    # 샘플 쿼리들 - 다양한 쇼핑 검색 시나리오
    sample_queries = [
        {
            "inputs": {"query": "남자 셔츠 추천"},
            "outputs": {"expected_info": "남성용 셔츠 상품 정보, 브랜드, 가격"}
        },
        # {
        #     "inputs": {"query": "원피스 추천"},
        #     "outputs": {"expected_info": "여성용 원피스 상품 정보, 스타일, 가격"}
        # },
        # {
        #     "inputs": {"query": "운동화 추천"},
        #     "outputs": {"expected_info": "운동화 상품 정보, 브랜드, 가격, 사이즈"}
        # },
        # {
        #     "inputs": {"query": "겨울 패딩 추천"},
        #     "outputs": {"expected_info": "겨울 패딩 상품 정보, 브랜드, 보온성, 가격"}
        # },
        # {
        #     "inputs": {"query": "청바지 추천"},
        #     "outputs": {"expected_info": "청바지 상품 정보, 핏, 브랜드, 가격"}
        # }
    ]
    
    try:
        # 기존 데이터셋이 있는지 확인
        try:
            dataset = client.read_dataset(dataset_name=dataset_name)
            print(f"기존 데이터셋 사용: {dataset_name}")
            return dataset.id
        except:
            pass
        
        # 새 데이터셋 생성
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="스크래핑 에이전트 평가용 쇼핑 쿼리 데이터셋"
        )
        
        # 샘플 데이터 추가
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[item["inputs"] for item in sample_queries],
            outputs=[item["outputs"] for item in sample_queries]
        )
        
        print(f"새 데이터셋 생성: {dataset_name} (ID: {dataset.id})")
        return dataset.id
        
    except Exception as e:
        print(f"데이터셋 생성 중 오류 발생: {e}")
        raise


async def main():
    """
    메인 평가 실행 함수
    
    1. LangSmith 클라이언트 초기화
    2. 평가용 데이터셋 준비
    3. 스크래핑 에이전트 평가 실행
    4. 결과 출력
    """
    # 환경 변수 확인
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not langsmith_api_key:
        print("오류: LANGSMITH_API_KEY 환경변수가 설정되지 않았습니다.")
        return
    
    if not openai_api_key:
        print("오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        return
    
    # LangSmith 클라이언트 초기화
    client = Client(api_key=langsmith_api_key)
    
    # 평가용 데이터셋 준비
    dataset_name = "scraping_agent_evaluation"
    try:
        dataset_id = create_sample_dataset(client, dataset_name)
    except Exception as e:
        print(f"데이터셋 준비 실패: {e}")
        return
    
    # 평가자 리스트 가져오기
    evaluators = get_scraping_evaluators()
    
    # 평가 실행 설정
    experiment_name = f"scraping_agent_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"평가 시작: {experiment_name}")
    print(f"데이터셋: {dataset_name}")
    print(f"평가자 수: {len(evaluators)}")
    
    try:
        # LangSmith 평가 실행
        # 주의: run_scraping_agent 함수를 동기 함수로 래핑해야 합니다
        def sync_run_scraping_agent(inputs):
            return asyncio.run(run_scraping_agent(inputs))
        
        results = evaluate(
            sync_run_scraping_agent,  # 평가할 함수
            data=dataset_name,        # 데이터셋 이름 또는 ID
            evaluators=evaluators,    # 평가자 리스트
            experiment_prefix=experiment_name,  # 실험 이름
            max_concurrency=1,        # 동시 실행 수 (에이전트 부하 고려)
            description="스크래핑/크롤링 에이전트 성능 평가"
        )
        
        print(f"\n평가 완료!")
        print(f"실험 이름: {experiment_name}")
        print(f"평가 결과는 LangSmith 대시보드에서 확인하세요: https://smith.langchain.com/")
        
        # 간단한 결과 요약 출력
        if hasattr(results, 'results'):
            total_runs = len(results.results)
            print(f"총 평가 실행 수: {total_runs}")
            
            # 평균 점수 계산 (모든 평가자 점수 합계)
            all_scores = []
            for result in results.results:
                if hasattr(result, 'evaluation_results'):
                    total_score = sum(eval_result.score for eval_result in result.evaluation_results)
                    all_scores.append(total_score)
            
            if all_scores:
                avg_score = sum(all_scores) / len(all_scores)
                print(f"평균 총점: {avg_score:.1f}/100 (4개 평가자 합계)")
        
    except Exception as e:
        print(f"평가 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


def run_single_evaluation():
    """
    단일 쿼리에 대한 평가 테스트
    
    전체 데이터셋 평가 전에 단일 쿼리로 테스트해볼 수 있습니다.
    """
    import asyncio
    
    async def test_single():
        # 테스트 쿼리
        test_input = {"query": "남자 셔츠 추천"}
        
        print("단일 쿼리 테스트 시작...")
        print(f"쿼리: {test_input['query']}")
        
        # 에이전트 실행
        result = await run_scraping_agent(test_input)
        
        print(f"\n실행 결과:")
        print(f"실행 시간: {result['execution_time']:.2f}초")
        print(f"최종 답변 길이: {len(result['final_answer'])} 문자")
        print(f"에이전트 메시지 수: {len(result['agent_messages'])}")
        print(f"최종 답변 미리보기: {result['final_answer'][:200]}...")
        
        return result
    
    return asyncio.run(test_single())


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 단일 쿼리 테스트 모드
        run_single_evaluation()
    else:
        # 전체 평가 모드
        asyncio.run(main())