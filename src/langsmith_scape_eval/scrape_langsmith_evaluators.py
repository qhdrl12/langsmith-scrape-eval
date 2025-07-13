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
- run.outputs.messages: LangGraph 표준 메시지 배열
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



# =====================================================================================
# 기본 모델과 유틸리티 함수들
# =====================================================================================

class EvaluationResponse(BaseModel):
    """LLM 평가 응답 모델"""
    score: float
    reason: str


def extract_tool_calls_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    LangGraph messages에서 도구 호출 정보를 추출
    
    Args:
        messages: LangGraph의 표준 메시지 배열
        
    Returns:
        추출된 도구 호출 정보 리스트 (tool_name, input_args, output_result, tool_status 포함)
    """
    tool_calls = []
    tool_call_map = {}
    
    # 1단계: 도구 호출 요청 수집
    for msg in messages:
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
    for msg in messages:
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
        messages = run.outputs.get("messages", [])
        
        if not query or not messages:
            return {"score": 0, "reason": "필수 데이터가 누락되었습니다."}
        
        # 도구 호출 정보 추출 및 분석
        tool_calls = extract_tool_calls_from_messages(messages)
        if not tool_calls:
            return {"score": 0, "reason": "도구 호출이 없습니다."}
        
        print(f"🔧 도구 호출: {len(tool_calls)}개")
        
        # LLM 평가 프롬프트
        prompt = f"""
스크래핑 에이전트의 도구 선택 및 호출 전략을 평가해주세요.

**사용자 질문**: {query}

**도구 호출 데이터**:
{json.dumps(tool_calls, ensure_ascii=False, indent=2)}

**평가 기준 (총 100점)**:
1. **쇼핑 도메인 적합성** (30점): 무신사 등 쇼핑몰 검색에 적절한 도구/URL 선택
2. **검색 전략의 정확성** (30점): 사용자 쿼리에 맞는 카테고리/키워드로 올바른 URL 구성
3. **호출 효율성** (20점): 중복 호출 방지, 논리적 탐색 순서
4. **다각도 정보 수집** (20점): 상품 정보를 위한 다양한 접근 방식 시도

**스크래핑 특화 채점 기준**:
- 쇼핑몰이 아닌 사이트 접근: 도메인 적합성 0점
- 잘못된 상품 카테고리나 검색어: 전략 정확성 0점  
- 동일 URL 반복 호출: 효율성 0점
- 상품 정보와 무관한 페이지 접근: 정보 수집 0점
- 검색 결과가 아닌 메인 페이지만 접근: 전략 정확성 최대 50%

**보너스 요소**:
- 카테고리별 다양한 검색 시도: +5점
- 필터링이나 정렬 옵션 활용: +5점

0-100점 사이의 점수와 구체적인 근거를 제시해주세요.
        """
        
        score, reason = evaluate_with_llm(prompt, "도구선택평가")
        
        return {
            "score": score,
            "reason": reason,
            "metadata": {
                "tool_count": len(tool_calls)
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
        messages = run.outputs.get("messages", [])
        
        if not query or not messages:
            return {"score": 0, "reason": "필수 데이터가 누락되었습니다."}
        
        # 도구 호출 정보 추출 및 분석
        tool_calls = extract_tool_calls_from_messages(messages)
        if not tool_calls:
            return {"score": 0, "reason": "도구 호출이 없습니다."}
        
        print(f"🚀 도구 실행: {len(tool_calls)}개")
        
        # LLM 평가 프롬프트
        prompt = f"""
스크래핑 도구 실행의 품질과 성공률을 평가해주세요.

**사용자 질문**: {query}

**도구 실행 데이터**:
{json.dumps(tool_calls, ensure_ascii=False, indent=2)}


**평가 기준 (총 100점)**:
1. **스크래핑 성공률** (40점): 웹페이지가 정상적으로 로드되고 파싱되었는가?
2. **상품 데이터 수집 품질** (30점): 실제 상품 정보(이름, 가격, 브랜드 등)를 수집했는가?
3. **오류 대응력** (20점): 접근 거부, 404 등의 오류에 적절히 대응했는가?
4. **데이터 완성도** (10점): 수집된 HTML/JSON이 완전하고 구조화되어 있는가?

**스크래핑 특화 채점 기준**:
- 403/404/500 등 HTTP 오류: 성공률 대폭 감점
- 빈 페이지나 로딩 실패: 품질 0점
- 상품 목록이 아닌 일반 페이지 수집: 품질 최대 30%
- 반복적인 접근 실패 시 대안 시도 없음: 대응력 0점
- 로봇 차단이나 보안 정책으로 인한 차단: 대응력 감점

**보너스 요소**:
- 차단 회피를 위한 헤더 설정이나 딜레이 적용: +5점
- 여러 페이지나 카테고리에서 안정적인 수집: +5점

0-100점 사이의 점수와 구체적인 근거를 제시해주세요.
        """
        
        score, reason = evaluate_with_llm(prompt, "도구실행평가")
        
        return {
            "score": score,
            "reason": reason,
            "metadata": {
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
        messages = run.outputs.get("messages", [])
        
        if not query or not messages:
            return {"score": 0, "reason": "필수 데이터가 누락되었습니다."}
        
        # 도구 호출 정보 추출
        tool_calls = extract_tool_calls_from_messages(messages)
        if not tool_calls:
            return {"score": 0, "reason": "도구 호출이 없습니다."}
        
        print(f"📊 데이터 추출: {len(tool_calls)}개")
        
        # LLM 평가 프롬프트
        prompt = f"""
스크래핑으로 수집된 상품 데이터의 품질과 추출 효과성을 평가해주세요.

**사용자 질문**: {query}

**추출 데이터**:
{json.dumps(tool_calls, ensure_ascii=False, indent=2)}


**평가 기준 (총 100점)**:
1. **상품 정보 관련성** (40점): 질문한 상품 유형과 일치하는 데이터를 추출했는가?
2. **상품 데이터 완성도** (30점): 상품명, 가격, 브랜드, 이미지 등 핵심 정보가 포함되었는가?
3. **데이터 구조화 품질** (20점): HTML에서 구조화된 상품 정보를 잘 추출했는가?
4. **정보 정확성** (10점): 추출된 상품 정보가 정확하고 신뢰할 수 있는가?

**쇼핑 스크래핑 특화 채점 기준**:
- 검색한 상품과 다른 카테고리 데이터: 관련성 0점
- 상품 목록이 없거나 일반 페이지 내용만: 완성도 0점
- 단순 HTML 덤프 (상품 정보 추출 없음): 구조화 0-10점
- 품절 상품만 수집하거나 가격 정보 누락: 정확성 감점
- 중복된 상품이나 광고성 콘텐츠: 완성도 감점

**고품질 상품 데이터 보너스**:
- 다양한 브랜드/가격대 상품 수집: +5점
- 상품 상세 정보(리뷰, 평점, 옵션 등) 포함: +5점
- 할인 정보나 재고 상태 정보 포함: +3점

0-100점 사이의 점수와 구체적인 근거를 제시해주세요.
        """
        
        score, reason = evaluate_with_llm(prompt, "데이터추출평가")
        
        return {
            "score": score,
            "reason": reason,
            "metadata": {
                "total_tools": len(tool_calls)
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
스크래핑 에이전트의 최종 상품 추천 답변 품질을 평가해주세요.

**사용자 질문**: {query}

**최종 답변**:
{final_answer}

**평가 기준 (총 100점)**:
1. **쇼핑 질문 적합성** (30점): 요청한 상품에 대한 구체적인 추천을 제공하는가?
2. **상품 정보 유용성** (25점): 상품명, 브랜드, 가격 등 구매 결정에 필요한 정보를 제공하는가?
3. **추천 답변 완성도** (25점): 여러 옵션 제시, 비교 정보, 구매 가이드 등이 포함되었는가?
4. **실용적 도움도** (20점): 실제 구매로 이어질 수 있는 실질적인 도움을 제공하는가?

**쇼핑 추천 특화 감점 요소**:
- "죄송합니다, 찾을 수 없습니다" 식 회피: 적합성 0점
- "직접 검색해보세요" 같은 책임 전가: 도움도 0점
- 상품명 없이 일반적 쇼핑 조언만: 유용성 0-15점
- 가격이나 브랜드 정보 없는 추천: 완성도 감점
- 품절/단종 상품만 추천: 실용성 감점
- 요청과 다른 카테고리 상품 추천: 적합성 대폭 감점

**고품질 쇼핑 추천 보너스**:
- 구체적인 상품명과 브랜드로 3개 이상 추천: +10점
- 가격대별 또는 스타일별 분류된 추천: +8점
- 할인 정보나 구매 팁 제공: +5점
- 무신사 링크나 구매 방법 안내: +3점

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