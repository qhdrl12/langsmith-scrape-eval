"""
Universal LangGraph Tool Evaluators for LangSmith

이 모듈은 LangGraph 에이전트의 도구 사용을 평가하는 범용 평가자들을 제공합니다.
LangSmith 평가 프레임워크와 호환되며, 다양한 도메인에서 재사용 가능합니다.

주요 평가 영역:
1. 도구 선택 및 호출의 적절성
2. 도구 실행 품질 및 성공률
3. 데이터 추출/수집의 효과성
4. 최종 답변의 품질과 유용성

입력 형식:
- run.outputs.agent_messages: LangGraph 표준 메시지 배열
- run.outputs.final_answer: 에이전트 최종 답변
- example.inputs.query: 사용자 질문
"""

import json
import os
from typing import Dict, Any, List, Optional
from langsmith.evaluation import run_evaluator
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# =====================================================================================
# 설정 상수들 (추후 설정 파일로 분리 가능)
# =====================================================================================

EVALUATION_CONFIG = {
    "max_score_per_evaluator": 100,  # 각 평가자당 최대 점수
    "llm_model": "gpt-4.1",
    "llm_temperature": 0.1,
}

# 도구 타입 분류 키워드 (확장 가능)
TOOL_TYPE_KEYWORDS = {
    "scraping": ["scrape", "extract", "parse"],
    "crawling": ["crawl", "fetch", "browse"],
    "search": ["search", "query", "find"],
    "api": ["api", "request", "call"],
}

# 에러 감지 키워드 (확장 가능)
ERROR_KEYWORDS = [
    "error", "오류", "failed", "실패", "not found", "찾을 수 없",
    "access denied", "접근 거부", "timeout", "시간 초과",
    "internal server error", "서버 오류", "dns resolution failed",
    "connection failed", "unsupported parameter", "invalid url",
    "forbidden", "403", "404", "500", "페이지를 찾을 수 없습니다"
]

# =====================================================================================
# 기본 모델과 유틸리티 함수들
# =====================================================================================

class EvaluationResponse(BaseModel):
    """LLM 평가 응답 모델"""
    score: float
    reason: str


def extract_tool_calls_from_messages(agent_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    LangGraph agent_messages에서 도구 호출 정보를 추출
    
    Args:
        agent_messages: LangGraph의 표준 메시지 배열
        
    Returns:
        추출된 도구 호출 정보 리스트 (tool_name, input_args, output_result, tool_status 포함)
    """
    tool_calls = []
    tool_call_map = {}
    
    # 1단계: 도구 호출 요청 수집
    for msg in agent_messages:
        if msg.get('tool_calls'):
            for tool_call in msg['tool_calls']:
                tool_id = tool_call.get('id', '')
                if tool_id:
                    tool_call_map[tool_id] = {
                        'tool_name': tool_call.get('name', 'unknown_tool'),
                        'input_args': tool_call.get('args', {}),
                        'output_result': "",
                        'tool_status': "called"
                    }
    
    # 2단계: 도구 실행 결과 수집
    for msg in agent_messages:
        if msg.get('type') == 'tool':
            tool_id = msg.get('tool_call_id', '')
            content = msg.get('content', '')
            
            if tool_id and tool_id in tool_call_map:
                tool_call_map[tool_id]['output_result'] = content
                tool_call_map[tool_id]['tool_status'] = 'success' if content else 'failed'
            else:
                # ID 매칭 실패한 경우
                tool_calls.append({
                    'tool_name': msg.get('name', 'unknown_tool'),
                    'input_args': {},
                    'output_result': content,
                    'tool_status': 'success' if content else 'failed'
                })
    
    # 3단계: 매핑된 도구 호출들 추가
    tool_calls.extend(tool_call_map.values())
    return tool_calls


def get_llm_evaluator() -> ChatOpenAI:
    """LLM 평가자 인스턴스 생성"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    
    return ChatOpenAI(
        model=EVALUATION_CONFIG["llm_model"],
        temperature=EVALUATION_CONFIG["llm_temperature"],
        api_key=api_key,
    )


def evaluate_with_llm(prompt: str, evaluator_name: str) -> tuple[float, str]:
    """공통 LLM 평가 로직"""
    try:
        llm = get_llm_evaluator()
        structured_llm = llm.with_structured_output(EvaluationResponse)
        response = structured_llm.invoke(prompt)
        
        max_score = EVALUATION_CONFIG["max_score_per_evaluator"]
        final_score = min(max(response.score, 0), max_score)
        
        print(f"✅ {evaluator_name}: {final_score}/{max_score}점")
        return final_score, response.reason
        
    except Exception as e:
        print(f"❌ {evaluator_name} LLM 오류: {str(e)}")
        return 0.0, f"평가 중 오류 발생: {str(e)}"


def classify_tools_by_type(tool_calls: List[Dict[str, Any]]) -> Dict[str, int]:
    """도구를 타입별로 분류하여 개수 반환"""
    counts = {tool_type: 0 for tool_type in TOOL_TYPE_KEYWORDS.keys()}
    counts["other"] = 0
    
    for tool_call in tool_calls:
        tool_name = tool_call['tool_name'].lower()
        classified = False
        
        for tool_type, keywords in TOOL_TYPE_KEYWORDS.items():
            if any(keyword in tool_name for keyword in keywords):
                counts[tool_type] += 1
                classified = True
                break
        
        if not classified:
            counts["other"] += 1
    
    return counts


def has_errors_in_output(output: str) -> bool:
    """출력에서 에러 감지"""
    return any(error_keyword in output.lower() for error_keyword in ERROR_KEYWORDS)


# =====================================================================================
# 평가자들
# =====================================================================================

@run_evaluator
def tool_selection_evaluator(run, example) -> Dict[str, Any]:
    """
    도구 선택 및 호출 적절성 평가자 (100점 만점)
    
    사용자 질문에 대한 도구 선택의 적절성, URL 구성의 정확성, 호출 효율성을 평가합니다.
    """
    try:
        # 기본 데이터 추출
        query = example.inputs.get("query", "")
        agent_messages = run.outputs.get("agent_messages", [])
        
        if not query or not agent_messages:
            return {"score": 0, "reason": "필수 데이터가 누락되었습니다."}
        
        # 도구 호출 정보 추출 및 분석
        tool_calls = extract_tool_calls_from_messages(agent_messages)
        if not tool_calls:
            return {"score": 0, "reason": "도구 호출이 없습니다."}
        
        tool_type_counts = classify_tools_by_type(tool_calls)
        
        print(f"🔧 도구 호출: {len(tool_calls)}개 ({tool_type_counts})")
        
        # LLM 평가 프롬프트
        prompt = f"""
사용자 질문에 대한 도구 선택 및 호출의 적절성을 평가해주세요.

**사용자 질문**: {query}

**도구 호출 데이터**:
{json.dumps(tool_calls, ensure_ascii=False, indent=2)}

**평가 기준 (총 100점)**:
1. **도구 선택 적절성** (30점): 질문에 맞는 올바른 도구 선택
2. **URL/파라미터 정확성** (30점): 올바른 URL 구성 및 파라미터 설정
3. **호출 효율성** (20점): 중복 호출 방지 및 논리적 순서
4. **전략적 다양성** (20점): 다각도 정보 수집 전략

**엄격한 채점 기준**:
- 질문과 무관한 도구/사이트: 해당 영역 0점
- 완전히 틀린 URL이나 카테고리: 해당 영역 0점  
- 동일한 도구+URL 중복 호출: 효율성 0점
- 모든 도구 실행 실패: 최대 30점

0-100점 사이의 점수와 구체적인 근거를 제시해주세요.
        """
        
        score, reason = evaluate_with_llm(prompt, "도구선택평가")
        
        return {
            "score": score,
            "reason": reason,
            "metadata": {
                "tool_count": len(tool_calls),
                "tool_types": tool_type_counts
            }
        }
        
    except Exception as e:
        return {"score": 0, "reason": f"평가 중 오류: {str(e)}"}


@run_evaluator
def tool_execution_evaluator(run, example) -> Dict[str, Any]:
    """
    도구 실행 품질 평가자 (100점 만점)
    
    도구 실행의 성공률, 오류 처리, 데이터 수집 품질을 평가합니다.
    """
    try:
        # 기본 데이터 추출
        query = example.inputs.get("query", "")
        agent_messages = run.outputs.get("agent_messages", [])
        
        if not query or not agent_messages:
            return {"score": 0, "reason": "필수 데이터가 누락되었습니다."}
        
        # 도구 호출 정보 추출 및 분석
        tool_calls = extract_tool_calls_from_messages(agent_messages)
        if not tool_calls:
            return {"score": 0, "reason": "도구 호출이 없습니다."}
        
        # 실행 통계 계산
        status_counts = {}
        error_count = 0
        
        for tool_call in tool_calls:
            status = tool_call['tool_status']
            status_counts[status] = status_counts.get(status, 0) + 1
            
            if has_errors_in_output(tool_call['output_result']):
                error_count += 1
        
        print(f"🚀 실행 상태: {status_counts}, 오류: {error_count}/{len(tool_calls)}개")
        
        # LLM 평가 프롬프트
        prompt = f"""
도구 실행의 품질과 성공률을 평가해주세요.

**사용자 질문**: {query}

**도구 실행 데이터**:
{json.dumps(tool_calls, ensure_ascii=False, indent=2)}

**실행 통계**: 상태 분포 {status_counts}, 오류 {error_count}/{len(tool_calls)}개

**평가 기준 (총 100점)**:
1. **실행 성공률** (40점): 도구가 정상적으로 실행되었는가?
2. **데이터 수집 품질** (30점): 의미있는 데이터를 수집했는가?
3. **오류 처리** (20점): 오류가 적고 적절히 처리되었는가?
4. **결과 완성도** (10점): 수집된 데이터의 완성도

**엄격한 채점 기준**:
- "called" 상태(실행 미완료): 성공률 0점
- 오류 메시지만 있는 경우: 해당 영역 0점
- 빈 결과나 무의미한 데이터: 품질 0-20점

0-100점 사이의 점수와 구체적인 근거를 제시해주세요.
        """
        
        score, reason = evaluate_with_llm(prompt, "도구실행평가")
        
        return {
            "score": score,
            "reason": reason,
            "metadata": {
                "status_distribution": status_counts,
                "error_count": error_count,
                "total_tools": len(tool_calls)
            }
        }
        
    except Exception as e:
        return {"score": 0, "reason": f"평가 중 오류: {str(e)}"}


@run_evaluator
def data_extraction_evaluator(run, example) -> Dict[str, Any]:
    """
    데이터 추출 효과성 평가자 (100점 만점)
    
    수집된 데이터의 품질, 관련성, 구조화 정도를 평가합니다.
    """
    try:
        # 기본 데이터 추출
        query = example.inputs.get("query", "")
        agent_messages = run.outputs.get("agent_messages", [])
        
        if not query or not agent_messages:
            return {"score": 0, "reason": "필수 데이터가 누락되었습니다."}
        
        # 도구 호출 정보 추출
        tool_calls = extract_tool_calls_from_messages(agent_messages)
        if not tool_calls:
            return {"score": 0, "reason": "도구 호출이 없습니다."}
        
        # 데이터 추출 통계
        total_data_length = sum(len(tc['output_result']) for tc in tool_calls)
        successful_extractions = sum(1 for tc in tool_calls 
                                   if tc['tool_status'] == 'success' and 
                                   len(tc['output_result']) > 0 and 
                                   not has_errors_in_output(tc['output_result']))
        
        print(f"📊 데이터 추출: {successful_extractions}/{len(tool_calls)}개 성공, 총 {total_data_length:,} 문자")
        
        # LLM 평가 프롬프트
        prompt = f"""
수집된 데이터의 품질과 추출 효과성을 평가해주세요.

**사용자 질문**: {query}

**추출 데이터**:
{json.dumps(tool_calls, ensure_ascii=False, indent=2)}

**추출 통계**: {successful_extractions}/{len(tool_calls)}개 성공, 총 {total_data_length:,} 문자

**평가 기준 (총 100점)**:
1. **데이터 관련성** (40점): 질문과 관련된 유용한 데이터가 추출되었는가?
2. **데이터 완성도** (30점): 충분하고 의미있는 정보가 포함되어 있는가?
3. **구조화 품질** (20점): 데이터가 잘 구조화되어 있는가?
4. **정확성** (10점): 추출된 정보가 정확한가?

**엄격한 채점 기준**:
- 질문과 무관한 데이터: 관련성 0점
- 오류 메시지나 빈 데이터: 완성도 0점
- 원시 HTML만 있는 경우: 구조화 0-10점
- 길이보다는 실제 유용성으로 평가

0-100점 사이의 점수와 구체적인 근거를 제시해주세요.
        """
        
        score, reason = evaluate_with_llm(prompt, "데이터추출평가")
        
        return {
            "score": score,
            "reason": reason,
            "metadata": {
                "successful_extractions": successful_extractions,
                "total_data_length": total_data_length,
                "extraction_rate": successful_extractions / len(tool_calls) if tool_calls else 0
            }
        }
        
    except Exception as e:
        return {"score": 0, "reason": f"평가 중 오류: {str(e)}"}


@run_evaluator
def answer_quality_evaluator(run, example) -> Dict[str, Any]:
    """
    최종 답변 품질 평가자 (100점 만점)
    
    최종 답변의 품질, 유용성, 완성도를 평가합니다.
    """
    try:
        # 기본 데이터 추출
        query = example.inputs.get("query", "")
        final_answer = run.outputs.get("final_answer", "")
        
        if not query:
            return {"score": 0, "reason": "질문이 제공되지 않았습니다."}
        
        if not final_answer:
            return {"score": 0, "reason": "최종 답변이 없습니다."}
        
        print(f"💬 최종 답변: {len(final_answer):,} 문자")
        
        # LLM 평가 프롬프트
        prompt = f"""
최종 답변의 품질과 유용성을 평가해주세요.

**사용자 질문**: {query}

**최종 답변**:
{final_answer}

**평가 기준 (총 100점)**:
1. **질문 적합성** (30점): 질문에 직접적으로 답변하고 있는가?
2. **정보 유용성** (25점): 실제로 도움이 되는 구체적인 정보를 제공하는가?
3. **답변 완성도** (25점): 충분히 상세하고 완성된 답변인가?
4. **사용자 도움도** (20점): 사용자의 문제 해결에 실질적으로 도움이 되는가?

**대폭 감점 요소**:
- 사과하며 회피하는 답변: 해당 영역 0점
- "다른 곳에서 확인하세요" 식의 책임 회피: 적합성 0점
- 일반론만 나열하고 구체적 정보 없음: 유용성 0-10점
- 짧고 성의없는 답변: 완성도 0-10점

**보너스 요소**:
- 구체적인 추천이나 비교 정보 제공: +10점
- 체계적이고 구조화된 답변: +5점

0-100점 사이의 점수와 구체적인 근거를 제시해주세요.
        """
        
        score, reason = evaluate_with_llm(prompt, "답변품질평가")
        
        return {
            "score": score,
            "reason": reason,
            "metadata": {
                "answer_length": len(final_answer),
                "has_structured_format": any(marker in final_answer for marker in ['##', '**', '|', '\n-', '\n*'])
            }
        }
        
    except Exception as e:
        return {"score": 0, "reason": f"평가 중 오류: {str(e)}"}


# =====================================================================================
# 평가자 그룹 함수
# =====================================================================================

def get_tool_evaluators() -> List:
    """
    범용 도구 평가자 리스트 반환
    
    각 평가자는 100점 만점으로 설계되어 있으며, 총 400점 만점입니다.
    LangSmith에서 개별 점수 및 평균을 자동 계산합니다.
    
    Returns:
        범용 도구 평가자 리스트
    """
    return [
        tool_selection_evaluator,    # 도구 선택 적절성 (100점)
        tool_execution_evaluator,    # 도구 실행 품질 (100점)
        data_extraction_evaluator,   # 데이터 추출 효과성 (100점)
        answer_quality_evaluator,    # 최종 답변 품질 (100점)
    ]


# 하위 호환성을 위한 별칭 (기존 코드와의 호환성)
get_scraping_evaluators = get_tool_evaluators