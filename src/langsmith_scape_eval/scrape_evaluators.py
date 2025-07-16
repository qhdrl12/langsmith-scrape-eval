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
from typing import Dict, Any, List
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
def search_results_evaluator(run, example) -> Dict[str, Any]:
    """
    검색 결과 품질 평가자 (100점 만점)
    
    사용자 질문에 대한 검색 결과의 관련성과 품질을 평가합니다.
    검색된 URL들이 질문과 얼마나 유사하고 적절한지를 평가합니다.
    """
    try:
        # 기본 데이터 추출
        question = example.inputs.get("question", "")
        messages = run.outputs.get("messages", [])
        
        # 디버깅 로그 추가
        print(f"🔍 검색결과평가 - 입력 데이터:")
        print(f"   - question: {question}")
        print(f"   - messages 수: {len(messages)}")
        print(f"   - example.inputs keys: {list(example.inputs.keys())}")
        print(f"   - run.outputs keys: {list(run.outputs.keys())}")
        
        if not question or not messages:
            return {"score": 0, "reason": "필수 데이터가 누락되었습니다."}
        
        # 도구 호출 정보 추출 및 분석
        tool_calls = extract_tool_calls_from_messages(messages)
        print(f"   - 전체 도구 호출 수: {len(tool_calls)}")
        if not tool_calls:
            return {"score": 0, "reason": "도구 호출이 없습니다."}
        
        # 검색 도구 호출만 필터링 (검색 관련 도구들)
        search_calls = [call for call in tool_calls if 'search' in call.get('tool_name', '').lower() or 'musinsa' in call.get('tool_name', '').lower()]
        print(f"   - 검색 도구 호출 수: {len(search_calls)}")
        
        # 도구 이름들 디버깅
        tool_names = [call.get('tool_name', 'unknown') for call in tool_calls]
        print(f"   - 도구 이름들: {tool_names}")
        
        print(f"🔍 검색 도구 호출: {len(search_calls)}개")
        
        # LLM 평가 프롬프트
        prompt = f"""
다음 사용자 질문에 대한 검색 결과의 품질과 관련성을 평가해주세요.

**사용자 질문**: {question}

**검색 도구 호출 데이터**:
{json.dumps(search_calls, ensure_ascii=False, indent=2)}

**평가 기준 (총 100점)**:

1. **검색 키워드 적합성** (30점)
   - 사용자 질문을 적절한 검색 키워드로 변환했는가?
   - 브랜드명, 카테고리, 스타일 등을 정확히 파악했는가?
   - 검색어가 너무 일반적이거나 지나치게 구체적이지 않은가?

2. **검색 결과 관련성** (50점)
   - 검색된 상품들이 사용자의 요구사항과 얼마나 부합하는가?
   - 카테고리, 브랜드, 가격대 등이 질문과 일치하는가?
   - 검색 결과가 실제로 존재하고 유의미한 상품인가?

3. **검색 결과 다양성** (20점)
   - 다양한 브랜드, 가격대, 스타일의 상품을 검색했는가?
   - 단, 사용자 질문에 특정 브랜드가 언급된 경우, 브랜드 다양성 부족으로 인한 감점은 없음
   - 사용자에게 선택의 폭을 제공할 수 있는 범위의 결과인가?
   - 특정 브랜드나 가격대에 편중되지 않았는가?

**점수 배분 가이드라인**:

검색 키워드 적합성 (30점) - 보수적 평가:
- 기본 점수: 0점 (검색 시도가 없으면 0점)
- 사용자 질문의 핵심 키워드 포함: +10점
- 브랜드명이나 카테고리 정확히 파악: +8점
- 검색어가 구체적이고 적절함: +7점
- 동의어나 관련어 활용: +5점
- 최대 30점, 실제 키워드 품질에 따라서만 가점

검색 결과 관련성 (50점) - 보수적 평가:
- 기본 점수: 0점 (검색 결과가 없으면 0점)
- 사용자 질문과 직접적으로 관련된 상품 발견: +25점
- 카테고리가 정확히 일치하는 상품들: +15점
- 브랜드나 스타일이 부합하는 상품들: +10점
- 상품 상세 정보(/app/goods/상품번호) 또는 상품 리뷰(/review/user)가 포함된 경우: 링크당 +(50 ÷ 전체 링크 개수)점
- 기타 링크(기획전, 스냅, 정보, 패션톡 등)는 점수 부여 안함 (0점)
- 최대 50점, 실제 관련성에 따라서만 가점

검색 결과 다양성 (20점) - 보수적 평가:
- 기본 점수: 0점 (단일 결과나 다양성이 없으면 0점)
- 서로 다른 브랜드의 상품들: +8점
- 다양한 가격대의 상품들: +6점
- 다양한 스타일/디자인의 상품들: +4점
- 사용자에게 선택의 폭 제공: +2점
- 최대 20점, 실제 다양성에 따라서만 가점

**점수 부여 조건 (보수적 평가)**:
- 검색 시도 없이는 점수 부여 안함
- 검색 결과가 없거나 오류 시 해당 항목 0점
- 의미있는 검색 결과가 있어야만 점수 부여
- 성공적인 검색 성과에 대해서만 해당 항목별 가점 부여
- 전체적으로 보수적 접근: 모호한 경우 낮은 점수 부여

**평가 시 고려사항**:
- 검색 도구의 제한사항을 고려하여 현실적으로 평가
- 사용자 의도를 정확히 파악했는지 중점 평가
- 검색 결과의 양보다는 질에 중점을 둠

0-100점 사이의 점수와 각 평가 기준별 상세한 근거를 제시해주세요.
        """
        
        score, reason = evaluate_with_llm(prompt, "검색결과평가")
        
        return {
            "score": score,
            "reason": reason,
            "metadata": {
                "search_calls_count": len(search_calls),
                "total_tools": len(tool_calls)
            }
        }
        
    except Exception as e:
        return {"score": 0, "reason": f"평가 중 오류: {str(e)}"}


@run_evaluator
def scraping_results_evaluator(run, example) -> Dict[str, Any]:
    """
    스크래핑 결과 품질 평가자 (100점 만점)
    
    웹사이트에서 스크래핑한 데이터의 품질과 정보 수집 완성도를 평가합니다.
    다양한 스크래핑 도구(크롤링, API 호출, 데이터 추출 등)를 포괄적으로 평가합니다.
    """
    try:
        # 기본 데이터 추출
        question = example.inputs.get("question", "")
        messages = run.outputs.get("messages", [])
        
        if not question or not messages:
            return {"score": 0, "reason": "필수 데이터가 누락되었습니다."}
        
        # 도구 호출 정보 추출 및 분석
        tool_calls = extract_tool_calls_from_messages(messages)
        if not tool_calls:
            return {"score": 0, "reason": "도구 호출이 없습니다."}
        
        # 스크래핑 도구 호출만 필터링
        scraping_calls = [call for call in tool_calls if 'scrape' in call.get('tool_name', '').lower()]
        
        print(f"🔧 스크래핑 도구 호출: {len(scraping_calls)}개")
        
        # LLM 평가 프롬프트
        prompt = f"""
다음 사용자 질문에 대한 스크래핑 결과의 품질과 데이터 수집 완성도를 평가해주세요.

**사용자 질문**: {question}

**스크래핑 도구 호출 데이터**:
{json.dumps(scraping_calls, ensure_ascii=False, indent=2)}

**평가 기준 (총 100점)**:

1. **스크래핑 성공률** (30점)
   - 웹페이지나 데이터 소스에 정상적으로 접근하고 데이터를 수집했는가?
   - HTTP 응답 코드가 200이며 실제 컨텐츠를 받았는가?
   - 접근 차단이나 로딩 실패 없이 데이터를 수집했는가?
   - **중요**: 단순한 HTTP 200이 아닌 실제 목표 정보가 포함된 의미있는 데이터인가?

2. **데이터 추출 품질** (40점)
   - 요청된 핵심 정보를 정확히 추출했는가?
   - 추출된 정보가 실제 웹페이지의 정보와 일치하는가?
   - 목표한 데이터의 핵심 속성들을 포함하고 있는가?

3. **데이터 완성도** (20점)
   - 수집된 정보가 완전하고 구조화되어 있는가?
   - 필수 정보가 모두 포함되어 있는가?
   - 추가 유용한 정보가 적절히 포함되어 있는가?

4. **오류 처리** (10점)
   - 접근 거부, 404 등의 오류에 적절히 대응했는가?
   - 실패 시 재시도나 대안 방법을 시도했는가?
   - 오류 상황을 명확히 파악하고 처리했는가?

**점수 배분 가이드라인**:

스크래핑 성공률 (30점) - 보수적 평가:
- 기본 점수: 0점 (아무것도 하지 않으면 0점)
- 데이터 소스 접근 성공 시: +5점
- 의미있는 목표 정보 추출 성공 시: +15점
- 다수 소스에서 일관된 성공 시: +10점
- 최대 30점, 실제 성과에 따라서만 가점

데이터 추출 품질 (40점) - 보수적 평가:
- 기본 점수: 0점 (추출된 정보가 없으면 0점)
- 핵심 정보 정확히 추출: +20점
- 부가 정보 정확히 추출: +10점
- 추가 속성들 추출: +5점
- 정보의 정확성과 완전성: +5점
- 최대 40점, 실제 추출 성과에 따라서만 가점

데이터 완성도 (20점) - 보수적 평가:
- 기본 점수: 0점 (구조화되지 않은 데이터는 0점)
- 데이터가 구조화되어 있음: +10점
- 필수 정보가 모두 포함: +5점
- 추가 유용한 정보들이 포함: +3점
- 데이터 형식이 일관되고 깔끔함: +2점
- 최대 20점, 실제 완성도에 따라서만 가점

오류 처리 (10점) - 보수적 평가:
- 기본 점수: 0점 (오류 처리가 없으면 0점)
- 오류 상황을 인식하고 있음: +3점
- 오류에 대한 적절한 대응: +4점
- 재시도나 대안 방법 시도: +2점
- 오류 메시지가 명확하고 유용함: +1점
- 최대 10점, 실제 오류 처리 능력에 따라서만 가점

**점수 부여 조건 (보수적 평가)**:
- 단순한 HTTP 200 응답만으로는 점수 부여 안함
- 실제 목표 데이터가 추출되어야만 점수 부여
- 오류나 실패 상황에서는 기본 0점
- 성공적인 결과에 대해서만 해당 항목별 가점 부여
- 전체적으로 보수적 접근: 잘못된 평가보다는 엄격한 평가 우선

**평가 시 고려사항**:
- **핵심**: HTTP 200 응답만으로는 성공이 아님. 실제 목표 데이터가 추출되어야 함
- 웹사이트의 접근 제한이나 기술적 한계를 고려하여 평가
- 수집된 데이터의 정확성과 완전성에 중점을 둠
- 스크래핑 도구의 안정성과 신뢰성을 평가
- 오류 발생 시 복구 능력과 대응 방법을 고려
- 의미있는 데이터 = 최소한 요청된 핵심 정보 중 일부는 포함되어야 함

0-100점 사이의 점수와 각 평가 기준별 상세한 근거를 제시해주세요.
        """
        
        score, reason = evaluate_with_llm(prompt, "스크래핑결과평가")
        
        return {
            "score": score,
            "reason": reason,
            "metadata": {
                "scraping_calls_count": len(scraping_calls),
                "total_tools": len(tool_calls)
            }
        }
        
    except Exception as e:
        return {"score": 0, "reason": f"평가 중 오류: {str(e)}"}




@run_evaluator
def answer_quality_evaluator(run, example) -> Dict[str, Any]:
    """
    최종 답변 품질 평가자 (100점 만점)
    
    크롤링 결과를 바탕으로 생성된 최종 답변의 품질, 관련성, 유용성을 평가합니다.
    """
    try:
        # 기본 데이터 추출
        question = example.inputs.get("question", "")
        final_answer = run.outputs.get("final_answer", "")
        
        if not question:
            return {"score": 0, "reason": "질문이 제공되지 않았습니다."}
        
        if not final_answer:
            return {"score": 0, "reason": "최종 답변이 없습니다."}
        
        print(f"💬 최종 답변: {len(final_answer):,} 문자")
        
        # LLM 평가 프롬프트
        prompt = f"""
다음 사용자 질문에 대한 크롤링 결과를 바탕으로 생성된 최종 답변의 품질을 평가해주세요.

**사용자 질문**: {question}

**최종 답변**:
{final_answer}

**평가 기준 (총 100점)**:

1. **질문 관련성** (40점)
   - 사용자 질문에 직접적으로 답변하고 있는가?
   - 요청한 상품 카테고리나 조건에 부합하는 답변인가?
   - 질문의 핵심 의도를 정확히 파악하고 대응했는가?

2. **정보 유용성** (35점)
   - 상품명, 브랜드, 가격 등 구매 결정에 필요한 정보를 제공하는가?
   - 제공된 정보가 정확하고 최신인가?
   - 사용자가 실제로 구매할 때 도움이 되는 정보인가?

3. **답변 완성도** (25점)
   - 추천 상품들이 구체적이고 충분히 제시되었는가?
   - 답변이 체계적으로 구성되어 있는가?
   - 필요한 정보가 누락되지 않았는가?

**점수 배분 가이드라인**:

질문 관련성 (40점) - 보수적 평가:
- 기본 점수: 0점 (답변이 없거나 관련 없으면 0점)
- 사용자 질문의 핵심 의도 파악: +20점
- 요청한 카테고리나 조건에 정확히 부합: +12점
- 질문에 직접적으로 답변: +8점
- 최대 40점, 실제 관련성에 따라서만 가점

정보 유용성 (35점) - 보수적 평가:
- 기본 점수: 0점 (유용한 정보가 없으면 0점)
- 상품명 정확히 제공: +15점
- 가격 정보 제공: +10점
- 브랜드나 구매처 정보 제공: +5점
- 추가 유용한 정보(링크, 특징 등) 제공: +5점
- 최대 35점, 실제 유용성에 따라서만 가점

답변 완성도 (25점) - 보수적 평가:
- 기본 점수: 0점 (불완전한 답변이면 0점)
- 추천 상품들이 구체적으로 제시: +12점
- 답변이 체계적으로 구성: +8점
- 필요한 정보가 빠짐없이 포함: +3점
- 읽기 쉽고 이해하기 쉬운 형태: +2점
- 최대 25점, 실제 완성도에 따라서만 가점

**점수 부여 조건 (보수적 평가)**:
- 회피적 답변이나 책임 전가 시 해당 항목 0점
- 일반적인 조언만 제공 시 유용성 0점
- 구체적인 상품 정보 없이는 완성도 0점
- 요청과 다른 카테고리 추천 시 관련성 0점
- 성공적인 답변 제공에 대해서만 해당 항목별 가점 부여
- 전체적으로 보수적 접근: 애매한 경우 낮은 점수 부여

**평가 시 고려사항**:
- 크롤링 데이터의 품질과 양을 고려하여 현실적으로 평가
- 답변의 길이보다는 내용의 질에 중점을 둠
- 사용자의 실제 구매 결정에 도움이 되는지 중점 평가
- 답변의 신뢰성과 정확성을 우선시

0-100점 사이의 점수와 각 평가 기준별 상세한 근거를 제시해주세요.
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
    에이전트 플로우 기반 평가자 리스트 반환
    
    각 평가자는 100점 만점으로 설계되어 있으며, 총 300점 만점입니다.
    에이전트 플로우: 질문 → 검색 → 크롤링 → 답변 생성
    
    Returns:
        에이전트 플로우 기반 평가자 리스트
    """
    return [
        search_results_evaluator,    # 1. 검색 결과 품질 평가 (100점)
        scraping_results_evaluator,  # 2. 스크래핑 결과 품질 평가 (100점)
        answer_quality_evaluator,    # 3. 최종 답변 품질 평가 (100점)
    ]


# 하위 호환성을 위한 별칭 (기존 코드와의 호환성)
get_scraping_evaluators = get_tool_evaluators