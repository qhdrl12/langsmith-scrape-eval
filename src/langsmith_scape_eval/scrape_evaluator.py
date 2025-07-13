import asyncio
import time
import json
import os
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from langgraph_sdk import get_client
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class EvaluationResponse(BaseModel):
    """LLM 평가 응답을 위한 구조화된 모델"""
    score: float
    reason: str
    improvements: str

@dataclass
class ToolCall:
    """도구 호출 정보"""
    tool_name: str
    input_args: Dict[str, Any]
    output_result: str
    tool_status: str

@dataclass
class InferenceOutput:
    """추론 실행 결과"""
    query: str
    session_id: str
    start_time: float
    end_time: float
    execution_duration: float
    final_answer: str = ""
    agent_messages: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save_to_file(self, file_path: str) -> None:
        """추론 결과를 JSON 파일로 저장
        
        Args:
            file_path: 저장할 파일 경로
        """
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 데이터를 JSON 직렬화 가능한 형태로 변환
        data = self.to_dict()
        data['saved_at'] = datetime.now().isoformat()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"추론 결과가 저장되었습니다: {file_path}")
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'InferenceOutput':
        """JSON 파일에서 추론 결과를 로드
        
        Args:
            file_path: 로드할 파일 경로
            
        Returns:
            InferenceOutput: 로드된 추론 결과
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # saved_at 필드 제거 (InferenceOutput에는 없는 필드)
        data.pop('saved_at', None)
        
        # InferenceOutput 객체 생성
        inference_output = cls(**data)
        
        print(f"추론 결과가 로드되었습니다: {file_path}")
        return inference_output


def extract_tool_calls_from_messages(agent_messages: List[Dict[str, Any]]) -> List[ToolCall]:
    """agent_messages에서 tool_calls 정보를 추출
    
    Args:
        agent_messages: agent 메시지 리스트
    
    Returns:
        List[ToolCall]: 추출된 tool_calls 리스트
    """
    tool_calls = []
    tool_call_map = {}  # ID별 도구 호출 매핑
    
    # 1단계: 도구 호출 요청 수집
    for msg in agent_messages:
        if msg.get('tool_calls'):
            for tool_call in msg['tool_calls']:
                tool_id = tool_call.get('id', '')
                if tool_id:
                    tool_call_map[tool_id] = ToolCall(
                        tool_name=tool_call.get('name', 'unknown_tool'),
                        input_args=tool_call.get('args', {}),
                        output_result="",
                        tool_status="called"
                    )
    
    # 2단계: 도구 실행 결과 수집 (type='tool'인 메시지)
    for msg in agent_messages:
        if msg.get('type') == 'tool':
            tool_id = msg.get('tool_call_id', '')
            tool_name = msg.get('name', 'unknown_tool')
            content = msg.get('content', '')
            
            if tool_id and tool_id in tool_call_map:
                # 기존 호출에 결과 업데이트
                tool_call_map[tool_id].output_result = content
                tool_call_map[tool_id].tool_status = 'success' if content else 'failed'
            else:
                # 새로운 도구 결과 (ID 매칭 실패)
                tool_calls.append(ToolCall(
                    tool_name=tool_name,
                    input_args={},
                    output_result=content,
                    tool_status='success' if content else 'failed'
                ))
    
    # 3단계: 매핑된 도구 호출들을 결과에 추가
    tool_calls.extend(tool_call_map.values())
    
    # 4단계: 최종 답변에서 상품 정보가 있다면 도구가 성공했다고 추정
    # (도구 실행 결과 메시지가 없는 경우에만 실행되는 fallback 로직)
    if tool_calls and all(tc.tool_status == "called" for tc in tool_calls):
        # 모든 도구가 "called" 상태이지만 최종 답변에 구체적 정보가 있는지 확인
        final_answer = ""
        for msg in agent_messages:
            if msg.get('content') and len(msg.get('content', '')) > 100:
                final_answer = msg.get('content', '')
                break
        
        # 최종 답변에 상품 정보가 있다면 도구가 성공했다고 추정 (fallback)
        if any(keyword in final_answer.lower() for keyword in ['가격', '원', '평점', '브랜드', '상품']):
            for tc in tool_calls:
                tc.tool_status = "success"
                tc.output_result = f"도구 실행 성공 (최종 답변에서 상품 정보 확인됨): {final_answer[:200]}..."
    
    return tool_calls


def get_llm_evaluator() -> ChatOpenAI:
    """LLM 평가자 인스턴스 생성"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY 환경변수가 설정되지 않았습니다. "
            "다음 명령으로 설정해주세요: export OPENAI_API_KEY='your-api-key'"
        )
    
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,  # 일관성 있는 평가를 위해 낮은 temperature
        api_key=api_key,
    )


def evaluate_tool_relevance_with_llm(tool_calls: List[ToolCall], query: str) -> Dict[str, Any]:
    """LLM을 사용하여 도구 요청의 질문 관련성 평가"""
    llm = get_llm_evaluator()
    
    # 도구 호출 정보를 상세하게 분석
    tool_summary = []
    url_counts = {}  # URL 중복 체크용
    
    for i, tool_call in enumerate(tool_calls, 1):
        args = tool_call.input_args
        url = args.get('url', 'URL 없음') if isinstance(args, dict) else str(args)
        
        # URL 중복 카운트
        if url in url_counts:
            url_counts[url] += 1
        else:
            url_counts[url] = 1
            
        tool_summary.append(f"{i}. {tool_call.tool_name}\n   - URL: {url}\n   - 상태: {tool_call.tool_status}")
    
    # 중복 URL 분석 - 도구별 분류
    duplicate_analysis = []
    for url, count in url_counts.items():
        if count > 1:
            # 해당 URL을 호출한 도구들 찾기
            tools_for_url = [tc.tool_name for tc in tool_calls if tc.input_args.get('url') == url]
            duplicate_analysis.append(f"{url} -> {tools_for_url} ({count}회)")
    
    if duplicate_analysis:
        duplicate_info = f"\n\n**중복 호출 분석**: {'; '.join(duplicate_analysis)}"
    else:
        duplicate_info = "\n\n**중복 호출**: 없음"
    
    tool_text = "\n".join(tool_summary) if tool_summary else "도구 호출 없음"
    
    prompt = f"""
사용자 질문에 대한 도구 호출의 관련성을 평가해주세요.

**사용자 질문**: {query}

**도구 호출 내역**:
{tool_text}{duplicate_info}

**평가 기준**:
1. **도구 선택 적절성**: 질문에 맞는 올바른 도구를 선택했는가?
2. **사이트 선택 적절성**: 질문과 관련된 적절한 웹사이트를 선택했는가?
3. **URL 구성의 정확성**: 원하는 정보를 얻기 위한 URL이 적절하게 구성되었는가?
4. **호출 효율성 및 전략성**: 도구 사용의 효율성과 전략적 접근
5. **전략적 다양성**: 다양한 정보 수집을 위한 전략적 접근이 있는가?

**중복 호출 평가 기준**:
- **완전히 동일한 도구 + 동일한 URL**: 큰 감점 (불필요한 중복)
- **다른 도구 + 동일한 URL**: 목적에 따라 평가
  * scraping + crawling 조합: 구조화된 데이터 + 전체 내용 수집이 모두 필요한 경우 적절
  * 하지만 하나의 도구로 충분한 정보 획득이 가능하다면 비효율적
- **동일한 도구 + 파라미터가 다른 URL**: 다양성으로 인정
- **질문과 전혀 관련없는 사이트 선택**: 큰 감점
"""
    
    structured_llm = llm.with_structured_output(EvaluationResponse)
    response = structured_llm.invoke(prompt)
    
    return {
        "score": min(max(response.score, 0), 25),
        "reason": response.reason,
        "improvements": response.improvements,
        "llm_response": f"점수: {response.score}, 이유: {response.reason}"
    }


def evaluate_crawling_quality_with_llm(tool_calls: List[ToolCall], query: str) -> Dict[str, Any]:
    """LLM을 사용하여 크롤링 품질 평가"""
    llm = get_llm_evaluator()
    
    # 크롤링 도구만 필터링
    crawling_tools = [tc for tc in tool_calls if 'crawl' in tc.tool_name.lower()]
    
    tool_summary = []
    
    for i, tool_call in enumerate(crawling_tools, 1):
        result_preview = tool_call.output_result[:400] + "..." if len(tool_call.output_result) > 400 else tool_call.output_result
        tool_summary.append(f"{i}. {tool_call.tool_name} (상태: {tool_call.tool_status})\n   - 결과 미리보기: {result_preview}")
    
    tool_text = "\n".join(tool_summary) if tool_summary else "크롤링 도구 사용 없음"
    
    prompt = f"""
사용자 질문에 대한 크롤링 도구 사용의 품질을 엄격하게 평가해주세요.

**사용자 질문**: {query}

**크롤링 도구 사용 결과**:
{tool_text}

**중요한 상태 설명**:
- "called": 도구 호출만 했을 뿐 아직 실행 결과가 없음 (실패로 간주)
- "success": 도구 실행이 완료됨 (하지만 내용을 봐야 실제 성공 여부 판단)
- "failed": 명백한 실패

**평가 기준**:
1. **실제 크롤링 성공 여부**: 결과에 실제 웹페이지 내용이 포함되어 있는가?
2. **내용의 유용성**: 사용자 질문에 답할 수 있는 의미있는 데이터가 있는가?
3. **오류 없음**: 오류 메시지나 접근 실패 메시지가 없는가?

**엄격한 채점 기준**:
- 상태가 "called"이면 무조건 0점 (실행 결과 없음)
- 결과가 비어있거나 "페이지를 찾을 수 없습니다" 등 오류 메시지만 있으면 0-3점
- 실제 상품 정보나 웹페이지 내용이 보여야 15점 이상 가능
- 질문과 관련된 유용한 정보가 풍부해야 20점 이상 가능

**중요**: 관대하게 평가하지 말고, 실제로 유용한 크롤링 결과가 있을 때만 높은 점수를 주세요.
"""
    
    structured_llm = llm.with_structured_output(EvaluationResponse)
    response = structured_llm.invoke(prompt)
    
    return {
        "score": min(max(response.score, 0), 25),
        "reason": response.reason,
        "improvements": response.improvements,
        "llm_response": f"점수: {response.score}, 이유: {response.reason}"
    }


def evaluate_scraping_quality_with_llm(tool_calls: List[ToolCall], query: str) -> Dict[str, Any]:
    """LLM을 사용하여 스크래핑 품질 평가"""
    llm = get_llm_evaluator()
    
    # 스크래핑 도구만 필터링
    scraping_tools = [tc for tc in tool_calls if 'scrape' in tc.tool_name.lower()]
    
    tool_summary = []
    
    for i, tool_call in enumerate(scraping_tools, 1):
        result_preview = tool_call.output_result[:400] + "..." if len(tool_call.output_result) > 400 else tool_call.output_result
        tool_summary.append(f"{i}. {tool_call.tool_name} (상태: {tool_call.tool_status})\n   - 결과 미리보기: {result_preview}")
    
    tool_text = "\n".join(tool_summary) if tool_summary else "스크래핑 도구 사용 없음"
    
    prompt = f"""
사용자 질문에 대한 스크래핑 도구 사용의 품질을 엄격하게 평가해주세요.

**사용자 질문**: {query}

**스크래핑 도구 사용 결과**:
{tool_text}

**중요한 상태 설명**:
- "called": 도구 호출만 했을 뿐 아직 실행 결과가 없음 (실패로 간주)
- "success": 도구 실행이 완료됨 (하지만 실제 내용을 봐야 성공 여부 판단)
- "failed": 명백한 실패

**평가 기준**:
1. **실제 데이터 추출 성공**: 결과에 실제 상품명, 가격, 브랜드 등이 포함되어 있는가?
2. **질문 관련성**: 추출된 데이터가 사용자 질문에 답할 수 있는 내용인가?
3. **데이터 완성도**: 구조화되고 의미있는 정보가 추출되었는가?

**엄격한 채점 기준**:
- 상태가 "called"이면 무조건 0점 (실행 결과 없음)
- "페이지를 찾을 수 없습니다", "오류", "접근 거부" 등만 있으면 0-2점
- 빈 내용이나 의미없는 HTML만 있으면 0-3점
- 실제 상품 정보(이름, 가격, 브랜드 등)가 보여야 15점 이상 가능
- 질문과 직접 관련된 풍부한 상품 데이터가 있어야 20점 이상 가능

**중요**: "success=True"라고 되어 있어도 실제 내용이 없으면 0점에 가깝게 주세요. 실제로 쇼핑에 도움이 될 정보가 있을 때만 높은 점수를 주세요.
"""
    
    structured_llm = llm.with_structured_output(EvaluationResponse)
    response = structured_llm.invoke(prompt)
    
    return {
        "score": min(max(response.score, 0), 25),
        "reason": response.reason,
        "improvements": response.improvements,
        "llm_response": f"점수: {response.score}, 이유: {response.reason}"
    }


def evaluate_answer_quality_with_llm(final_answer: str, query: str) -> Dict[str, Any]:
    """LLM을 사용하여 답변 품질 평가"""
    llm = get_llm_evaluator()
    
    prompt = f"""
사용자 질문에 대한 AI 에이전트의 최종 답변 품질을 평가해주세요.

**사용자 질문**: {query}

**AI 에이전트 답변**:
{final_answer}

**평가 기준**:
1. **질문 관련성**: 질문에 대한 직접적이고 적절한 답변을 제공하는가?
2. **구체성과 유용성**: 실제로 도움이 되는 구체적인 정보나 추천을 제공하는가?
3. **답변의 완성도**: 충분히 상세하고 완성된 답변인가?
4. **적극성**: 사용자 요청에 성의있게 답변하려고 노력했는가?

**대폭 감점 요소**:
- "죄송합니다"로 시작하면서 실질적 도움 없이 끝나는 회피성 답변
- "다른 사이트를 확인하세요"라며 책임 회피
- 사용자가 요청한 추천을 전혀 제공하지 않음
- 구체적인 상품명, 브랜드, 가격 등 유용한 정보 부재

**높은 점수 조건**:
- 실제 상품 추천 제공
- 구체적인 상품 정보 포함
- 사용자에게 실질적으로 도움이 되는 내용
"""
    
    structured_llm = llm.with_structured_output(EvaluationResponse)
    response = structured_llm.invoke(prompt)
    
    return {
        "score": min(max(response.score, 0), 25),
        "reason": response.reason,
        "improvements": response.improvements,
        "llm_response": f"점수: {response.score}, 이유: {response.reason}"
    }


@dataclass
class EvaluationResult:
    """평가 결과"""
    query: str
    crawling_score: float = 0.0      # 크롤링 실행 점수 (0-25점)
    scraping_score: float = 0.0      # 데이터 추출 품질 점수 (0-25점)
    tool_relevance_score: float = 0.0  # 도구 호출 관련성 점수 (0-25점)
    answer_quality_score: float = 0.0  # 최종 답변 품질 점수 (0-25점)
    total_score: float = 0.0
    reason: str = ""
    
    # LLM 기반 세부 평가 결과
    tool_relevance_details: Dict[str, Any] = field(default_factory=dict)
    crawling_quality_details: Dict[str, Any] = field(default_factory=dict)
    scraping_quality_details: Dict[str, Any] = field(default_factory=dict)
    answer_quality_details: Dict[str, Any] = field(default_factory=dict)


# 기존 하드코딩된 평가 함수들은 LLM 기반 평가로 대체됨


async def run_inference(query: str) -> InferenceOutput:
    """추론 실행 및 출력 생성
    
    LangGraph Agent를 실행하여 사용자 질문에 대한 답변과 도구 호출 결과를 수집합니다.
    
    Args:
        query: 사용자 질문
    
    Returns:
        InferenceOutput: 추론 실행 결과 (도구 호출 내역, 최종 답변 등)
    """
    base_url = "http://127.0.0.1:2024"
    client = get_client(url=base_url)

    # 새로운 스레드 생성
    thread = await client.threads.create()
    session_id = thread["thread_id"]
    start_time = time.time()
    
    # 추론 결과 객체 초기화
    inference_output = InferenceOutput(
        query=query,
        session_id=session_id,
        start_time=start_time,
        end_time=0.0,
        execution_duration=0.0
    )
    
    # Agent에 전달할 메시지 구성
    input_data = {
        "messages": [
            {"role": "system", "content": "너는 무신사 검색 전문가야. 사용자에 질문에 해당하는 물건을 검색해줘. 정보가 부족하더라도 최대한 잘 찾아야하며 사용자에게 추가 정보를 요구하면 안돼."},
            {"role": "user", "content": query}
        ]
    }

    
    # Agent 실행 및 스트림 처리
    async for chunk in client.runs.stream(
        thread["thread_id"],
        "agent",
        input=input_data,
        stream_mode="updates"
    ):
        if chunk.event == "updates":
            # Agent 메시지 처리
            if "agent" in chunk.data:
                messages = chunk.data.get("agent", {}).get("messages", [])
                for msg in messages:
                    print(f"[AGENT] msg content: {msg.get('content', '')}")
                    
                    # 모든 agent 메시지를 저장
                    inference_output.agent_messages.append(msg)
                    
                    # 내용이 있는 마지막 agent 메시지를 final_answer로 저장
                    if msg.get("content"):
                        inference_output.final_answer = msg.get("content", "")
            
            # 도구 실행 결과 메시지 처리 - agent_messages에 추가
            if "tools" in chunk.data:
                print(f"TOOL chunk_data: {chunk.data}")
                tool_messages = chunk.data.get("tools", {}).get("messages", [])
                for tool_msg in tool_messages:
                    tool_name = tool_msg.get("name", "unknown_tool")
                    tool_status = tool_msg.get("status", "unknown")
                    tool_call_id = tool_msg.get("tool_call_id", "no_id")
                    content = tool_msg.get("content", "")
                    
                    print(f"[TOOL] {tool_name} (status: {tool_status}, id: {tool_call_id})")
                    print(f"[TOOL] content: {content}")
                    print(f"[TOOL] Raw content length: {len(content)} chars")
                    
                    # 도구 실행 결과를 agent_messages에 추가
                    tool_result_msg = {
                        "type": "tool",
                        "name": tool_name,
                        "tool_call_id": tool_call_id,
                        "content": content,
                        "status": tool_status
                    }
                    inference_output.agent_messages.append(tool_result_msg)
    
    # 추론 완료 처리
    inference_output.end_time = time.time()
    inference_output.execution_duration = inference_output.end_time - inference_output.start_time
    
    return inference_output


def evaluate_inference(inference_output: InferenceOutput) -> EvaluationResult:
    """LLM을 사용하여 추론 결과를 평가
    
    스크래핑/크롤링 에이전트의 성능을 다음 4개 영역에서 LLM으로 평가:
    1. 도구 요청 관련성 점수 (25점): 도구 호출이 질문과 관련있는지
    2. 크롤링 품질 점수 (25점): 크롤링 도구 사용의 품질
    3. 스크래핑 품질 점수 (25점): 스크래핑 도구 사용의 품질
    4. 답변 품질 점수 (25점): 최종 답변의 품질과 관련성
    
    Args:
        inference_output: 추론 실행 결과
    
    Returns:
        EvaluationResult: LLM 기반 평가 점수와 상세 분석 결과
    """
    query = inference_output.query
    tool_calls = extract_tool_calls_from_messages(inference_output.agent_messages)
    final_answer = inference_output.final_answer
    
    print("LLM 기반 평가 시작...")
    
    # 1. 도구 요청 관련성 평가 (LLM)
    print("1. 도구 요청 관련성 평가 중...")
    tool_relevance_eval = evaluate_tool_relevance_with_llm(tool_calls, query)
    tool_relevance_score = tool_relevance_eval["score"]
    
    # 2. 크롤링 품질 평가 (LLM)
    print("2. 크롤링 품질 평가 중...")
    crawling_quality_eval = evaluate_crawling_quality_with_llm(tool_calls, query)
    crawling_score = crawling_quality_eval["score"]
    
    # 3. 스크래핑 품질 평가 (LLM)
    print("3. 스크래핑 품질 평가 중...")
    scraping_quality_eval = evaluate_scraping_quality_with_llm(tool_calls, query)
    scraping_score = scraping_quality_eval["score"]
    
    # 4. 답변 품질 평가 (LLM)
    print("4. 답변 품질 평가 중...")
    answer_quality_eval = evaluate_answer_quality_with_llm(final_answer, query)
    answer_quality_score = answer_quality_eval["score"]
    
    # 총점 계산 (100점 만점)
    total_score = tool_relevance_score + crawling_score + scraping_score + answer_quality_score
    
    # 평가 이유 생성 (LLM 평가 결과 기반)
    reasons = [
        f"도구 관련성: {tool_relevance_score:.1f}점 - {tool_relevance_eval['reason']}",
        f"크롤링 품질: {crawling_score:.1f}점 - {crawling_quality_eval['reason']}",
        f"스크래핑 품질: {scraping_score:.1f}점 - {scraping_quality_eval['reason']}",
        f"답변 품질: {answer_quality_score:.1f}점 - {answer_quality_eval['reason']}"
    ]
    
    reason = " | ".join(reasons)
    
    print("LLM 기반 평가 완료!")
    
    return EvaluationResult(
        query=query,
        crawling_score=crawling_score,
        scraping_score=scraping_score,
        tool_relevance_score=tool_relevance_score,
        answer_quality_score=answer_quality_score,
        total_score=total_score,
        reason=reason,
        tool_relevance_details=tool_relevance_eval,
        crawling_quality_details=crawling_quality_eval,
        scraping_quality_details=scraping_quality_eval,
        answer_quality_details=answer_quality_eval,
    )


async def run_inference_only(query: str, save_path: str = None) -> InferenceOutput:
    """추론만 실행하고 결과를 저장 (평가는 별도로 수행)
    
    Args:
        query: 사용자 질문
        save_path: 결과를 저장할 파일 경로 (선택사항)
    
    Returns:
        InferenceOutput: 추론 실행 결과
    """
    print(f"추론 시작: {query}")
    
    # 추론 실행
    inference_output = await run_inference(query)
    
    tool_calls = extract_tool_calls_from_messages(inference_output.agent_messages)
    
    print(f"\n=== 추론 결과 ===")
    print(f"질문: {inference_output.query}")
    print(f"실행 시간: {inference_output.execution_duration:.2f}초")
    print(f"도구 호출 수: {len(tool_calls)}")
    print(f"최종 답변: {inference_output.final_answer}")
        
    # 결과 저장
    if save_path:
        inference_output.save_to_file(save_path)
    else:
        # 기본 파일명으로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"inference_results/inference_{timestamp}.json"
        inference_output.save_to_file(default_path)
    
    return inference_output


def evaluate_only(file_path: str = None, inference_output: InferenceOutput = None) -> EvaluationResult:
    """저장된 추론 결과를 로드하여 평가만 수행
    
    Args:
        file_path: 추론 결과 파일 경로 (file_path 또는 inference_output 중 하나 필수)
        inference_output: 추론 결과 객체 (file_path 또는 inference_output 중 하나 필수)
    
    Returns:
        EvaluationResult: 평가 결과
    """
    if file_path and inference_output:
        raise ValueError("file_path와 inference_output 중 하나만 제공해야 합니다.")
    
    if not file_path and not inference_output:
        raise ValueError("file_path 또는 inference_output 중 하나는 제공해야 합니다.")
    
    # 추론 결과 로드
    if file_path:
        print(f"저장된 추론 결과 로드: {file_path}")
        inference_output = InferenceOutput.load_from_file(file_path)
    
    tool_calls = extract_tool_calls_from_messages(inference_output.agent_messages)
    
    print(f"\n=== 평가 대상 정보 ===")
    print(f"질문: {inference_output.query}")
    print(f"도구 호출 수: {len(tool_calls)}")
    print(f"최종 답변 길이: {len(inference_output.final_answer)} 문자")
    
    # 평가 수행
    evaluation_result = evaluate_inference(inference_output)
    
    # 평가 결과 출력
    print(f"\n=== 평가 결과 ===")
    print(f"크롤링 점수: {evaluation_result.crawling_score:.1f}/25")
    print(f"스크래핑 점수: {evaluation_result.scraping_score:.1f}/25")
    print(f"도구 관련성 점수: {evaluation_result.tool_relevance_score:.1f}/25")
    print(f"답변 품질 점수: {evaluation_result.answer_quality_score:.1f}/25")
    print(f"총점: {evaluation_result.total_score:.1f}/100")
    print(f"평가 이유: {evaluation_result.reason}")
    
    # LLM 기반 세부 분석 결과 출력
    print(f"\n=== LLM 기반 세부 분석 ===")
    print(f"도구 관련성 평가: {evaluation_result.tool_relevance_details['reason']}")
    if evaluation_result.tool_relevance_details.get('improvements'):
        print(f"  개선점: {evaluation_result.tool_relevance_details['improvements']}")
    
    print(f"크롤링 품질 평가: {evaluation_result.crawling_quality_details['reason']}")
    if evaluation_result.crawling_quality_details.get('improvements'):
        print(f"  개선점: {evaluation_result.crawling_quality_details['improvements']}")
    
    print(f"스크래핑 품질 평가: {evaluation_result.scraping_quality_details['reason']}")
    if evaluation_result.scraping_quality_details.get('improvements'):
        print(f"  개선점: {evaluation_result.scraping_quality_details['improvements']}")
    
    print(f"답변 품질 평가: {evaluation_result.answer_quality_details['reason']}")
    if evaluation_result.answer_quality_details.get('improvements'):
        print(f"  개선점: {evaluation_result.answer_quality_details['improvements']}")
    
    # LLM 평가로 통계 정보 대체됨
    
    return evaluation_result


def batch_evaluate(results_dir: str) -> List[EvaluationResult]:
    """저장된 여러 추론 결과를 일괄 평가
    
    Args:
        results_dir: 추론 결과 파일들이 저장된 디렉토리 경로
    
    Returns:
        List[EvaluationResult]: 평가 결과 리스트
    """
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {results_dir}")
    
    # JSON 파일들 찾기
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"평가할 JSON 파일이 없습니다: {results_dir}")
        return []
    
    print(f"배치 평가 시작: {len(json_files)}개 파일")
    
    evaluation_results = []
    
    for json_file in sorted(json_files):
        file_path = os.path.join(results_dir, json_file)
        try:
            print(f"\n{'='*50}")
            print(f"평가 중: {json_file}")
            
            evaluation_result = evaluate_only(file_path=file_path)
            evaluation_results.append(evaluation_result)
            
        except Exception as e:
            print(f"오류 발생 ({json_file}): {e}")
            continue
    
    # 배치 평가 요약
    print(f"\n{'='*50}")
    print(f"배치 평가 완료: {len(evaluation_results)}/{len(json_files)} 성공")
    
    if evaluation_results:
        avg_score = sum(er.total_score for er in evaluation_results) / len(evaluation_results)
        print(f"평균 점수: {avg_score:.1f}/100")
        
        # 상위/하위 결과
        sorted_results = sorted(evaluation_results, key=lambda x: x.total_score, reverse=True)
        print(f"최고 점수: {sorted_results[0].total_score:.1f} (질문: {sorted_results[0].query})")
        print(f"최저 점수: {sorted_results[-1].total_score:.1f} (질문: {sorted_results[-1].query})")
    
    return evaluation_results


async def main():
    """메인 실행 함수 - 추론 실행과 평가를 순차적으로 수행"""
    query = "원피스 추천"
    
    print(f"추론 시작: {query}")
    
    # 1. 추론 실행 - Agent 실행 및 도구 호출 결과 수집
    inference_output = await run_inference(query)
    
    tool_calls = extract_tool_calls_from_messages(inference_output.agent_messages)
    
    print(f"\n=== 추론 결과 ===")
    print(f"질문: {inference_output.query}")
    print(f"실행 시간: {inference_output.execution_duration:.2f}초")
    print(f"도구 호출 수: {len(tool_calls)}")
    print(f"최종 답변: {inference_output.final_answer}")
    
    # 2. 평가 수행 - 수집된 결과를 바탕으로 성능 평가
    evaluation_result = evaluate_inference(inference_output)
    
    print(f"\n=== 평가 결과 ===")
    print(f"크롤링 점수: {evaluation_result.crawling_score:.1f}/25")
    print(f"스크래핑 점수: {evaluation_result.scraping_score:.1f}/25")
    print(f"도구 관련성 점수: {evaluation_result.tool_relevance_score:.1f}/25")
    print(f"답변 품질 점수: {evaluation_result.answer_quality_score:.1f}/25")
    print(f"총점: {evaluation_result.total_score:.1f}/100")
    print(f"평가 이유: {evaluation_result.reason}")
    
    # LLM 기반 세부 분석 결과 출력
    print(f"\n=== LLM 기반 세부 분석 ===")
    print(f"도구 관련성 평가: {evaluation_result.tool_relevance_details['reason']}")
    if evaluation_result.tool_relevance_details.get('improvements'):
        print(f"  개선점: {evaluation_result.tool_relevance_details['improvements']}")
    
    print(f"크롤링 품질 평가: {evaluation_result.crawling_quality_details['reason']}")
    if evaluation_result.crawling_quality_details.get('improvements'):
        print(f"  개선점: {evaluation_result.crawling_quality_details['improvements']}")
    
    print(f"스크래핑 품질 평가: {evaluation_result.scraping_quality_details['reason']}")
    if evaluation_result.scraping_quality_details.get('improvements'):
        print(f"  개선점: {evaluation_result.scraping_quality_details['improvements']}")
    
    print(f"답변 품질 평가: {evaluation_result.answer_quality_details['reason']}")
    if evaluation_result.answer_quality_details.get('improvements'):
        print(f"  개선점: {evaluation_result.answer_quality_details['improvements']}")
    
    # LLM 평가로 통계 정보 대체됨
    
    # 도구별 상세 정보
    print(f"\n=== 도구 호출 상세 ===")
    for i, tool_call in enumerate(tool_calls):
        print(f"{i+1}. {tool_call.tool_name} ({tool_call.tool_status})")
        print(f"   결과 길이: {len(tool_call.output_result)} 문자")
        print()
    
    # JSON 형태로 출력 (외부 시스템 연동용)
    print(f"\n=== JSON 결과 ===")
    print("## 추론 결과:")
    print(json.dumps(inference_output.to_dict(), ensure_ascii=False, indent=2))
    print("\n## 평가 결과:")
    print(json.dumps(asdict(evaluation_result), ensure_ascii=False, indent=2))
    
    return inference_output, evaluation_result


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="LangSmith 스크래핑 에이전트 평가 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 모드 (추론 + 평가)
  python scrape_evaluator.py
  
  # 추론만 실행
  python scrape_evaluator.py --mode inference --query "원피스 추천" --save-path inference_result.json
  
  # 저장된 결과만 평가
  python scrape_evaluator.py --mode evaluate --file inference_result.json
  
  # 여러 결과 일괄 평가
  python scrape_evaluator.py --mode batch --dir inference_results/
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["default", "inference", "evaluate", "batch"],
        default="default",
        help="실행 모드: default(추론+평가), inference(추론만), evaluate(평가만), batch(일괄평가)"
    )
    
    parser.add_argument(
        "--query",
        default="원피스 추천",
        help="검색 질문 (inference 모드에서 사용)"
    )
    
    parser.add_argument(
        "--save-path", 
        help="추론 결과 저장 경로 (inference 모드에서 사용)"
    )
    
    parser.add_argument(
        "--file",
        help="평가할 추론 결과 파일 경로 (evaluate 모드에서 사용)"
    )
    
    parser.add_argument(
        "--dir",
        help="일괄 평가할 디렉토리 경로 (batch 모드에서 사용)"
    )
    
    return parser.parse_args()


async def main_cli():
    """CLI 기반 메인 함수"""
    args = parse_args()
    
    if args.mode == "default":
        # 기본 모드: 추론 + 평가
        await main()
        
    elif args.mode == "inference":
        # 추론만 실행
        await run_inference_only(args.query, args.save_path)
        
    elif args.mode == "evaluate":
        # 평가만 실행
        if not args.file:
            print("오류: evaluate 모드에서는 --file 인자가 필요합니다.")
            return
        evaluate_only(file_path=args.file)
        
    elif args.mode == "batch":
        # 일괄 평가
        if not args.dir:
            print("오류: batch 모드에서는 --dir 인자가 필요합니다.")
            return
        batch_evaluate(args.dir)


if __name__ == "__main__":
    asyncio.run(main_cli())
