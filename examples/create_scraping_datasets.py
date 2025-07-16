"""
쇼핑 에이전트 스크래핑 평가용 데이터셋 생성

무신사 쇼핑 에이전트 평가를 위한 간단한 데이터셋을 생성합니다.

사용 방법:
    python examples/create_scraping_datasets.py
"""

import asyncio
import sys
import os
from pathlib import Path


# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.langsmith_scape_eval.dataset_manager import DatasetManager
from dotenv import load_dotenv


# 환경 변수 로드
load_dotenv()

# class EvaluationCriterion(BaseModel):
#     """개별 평가 기준 (추상화된 버전)"""
#     reason: str = Field(..., description="평가 기준 설명/이유")
#     score: float = Field(..., ge=0.0, le=1.0, description="기대 점수 (0-1)")


# class QueryAnalysis(BaseModel):
#     """쿼리 분석 정보"""
#     category: str = Field(..., description="상품 카테고리")
#     brand: Optional[str] = Field(None, description="브랜드명 (있는 경우)")
#     attributes: List[str] = Field(default_factory=list, description="상품 속성들 (색상, 사이즈, 소재 등)")


# class EvaluationCriteria(BaseModel):
#     """종합 평가 기준"""
#     search_relevance: EvaluationCriterion
#     crawling_performance: EvaluationCriterion
#     response_quality: EvaluationCriterion


# class TestScenario(BaseModel):
#     """테스트 시나리오"""
#     scenario: str = Field(..., description="시나리오 이름")
#     description: str = Field(..., description="시나리오 설명")
#     expected_outcome: Optional[str] = Field(None, description="예상 결과")


# class EvaluationQuery(BaseModel):
#     """평가용 쿼리 데이터"""
#     query_id: str = Field(..., description="쿼리 고유 식별자")
#     query: str = Field(..., description="사용자 원본 질의")
#     query_analysis: QueryAnalysis
#     expected_search_keywords: List[str] = Field(..., description="예상 검색 키워드들")
#     evaluation_criteria: EvaluationCriteria
#     test_scenarios: List[TestScenario] = Field(..., description="테스트 시나리오들")


async def create_shopping_dataset() -> str:
        """Create a sample shopping/e-commerce dataset."""
        print("🛒 쇼핑 에이전트 데이터셋 생성")
        print("=" * 40)

        manager = DatasetManager()
        
        shopping_queries = [
            {
                "question": "버뮤다 팬츠",
            },
            {
                "question": "롱 원피스",
            },
            {
                "question": "홀가먼트 니트",
            },
            {
                "question": "살로몬 버킷햇",
            },
            {
                "question": "살로몬 로튀스 버킷햇",
            },
            {
                "question": "나시 롱 원피스",
            },
            {
                "question": "곰돌이 잠옷 세트",
            },
            {
                "question": "마크모크 통굽 샌들",
            },
            {
                "question": "크록스 지비츠",
            },
            {
                "question": "가나디 반팔 블랙",
            }
        ]
            
        shopping_id = manager.create_agent_dataset("shopping_agent_dataset_new", shopping_queries, "shopping")
        print(f"✅ 쇼핑 데이터셋 생성 완료: {shopping_id}")

        return shopping_id


async def main():
    """
    쇼핑 에이전트 평가용 데이터셋 생성
    """
    print("🛒 쇼핑 에이전트 평가 데이터셋 생성")
    print("=" * 40)
    
    # LangSmith API 키 확인
    if not os.getenv("LANGSMITH_API_KEY"):
        print("❌ LANGSMITH_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   .env 파일에 LANGSMITH_API_KEY를 설정해주세요.")
        return
    
    try:
        await create_shopping_dataset()
        print("\n✅ 쇼핑 에이전트 데이터셋 생성 완료!")
        
    except KeyboardInterrupt:
        print("\n👋 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    asyncio.run(main())