# 가상환경 활성화 activate.bat / 비활성화 deactivate.bat
# 설치:
# pip install python-dotenv
# pip install langchain
# pip install langchain-openai

# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
import os

# API KEY 정보로드
load_dotenv()

print(f"[API KEY]\n{os.environ['OPENAI_API_KEY']}")