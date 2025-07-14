"""
쇼핑 에이전트 스크래핑 평가용 데이터셋 생성

무신사 쇼핑 에이전트 평가를 위한 간단한 데이터셋을 생성합니다.

사용 방법:
    python examples/create_scraping_datasets.py
"""

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


def create_shopping_dataset():
    """
    쇼핑 에이전트 스크래핑 평가용 데이터셋 생성
    """
    print("🛒 쇼핑 에이전트 데이터셋 생성")
    print("=" * 40)
    
    manager = DatasetManager()
    
    try:
        shopping_id = manager.create_shopping_dataset()
        print(f"✅ 쇼핑 데이터셋 생성 완료: {shopping_id}")
        return shopping_id
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return None




def main():
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
        create_shopping_dataset()
        print("\n✅ 쇼핑 에이전트 데이터셋 생성 완료!")
        
    except KeyboardInterrupt:
        print("\n👋 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()