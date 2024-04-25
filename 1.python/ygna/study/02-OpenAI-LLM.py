from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import openai

load_dotenv()

# 0. 설치된 openai 버전 체크
print(openai.__version__)

# 1. ChatOpenAI
# 객체 생성
llm = ChatOpenAI(
    temperature=0.1,  # 창의성 (0.0 ~ 2.0)
    max_tokens=2048,  # 최대 토큰수
    model_name="gpt-3.5-turbo",  # 모델명
)

# 질의내용
question = "대한민국의 수도는 어디인가요?"

# 질의
print(f"[답변]: {llm.invoke(question)}")

# 2. 프롬프트 템플릿 활용
# 질문 템플릿 형식 정의
template = "{country}의 수도는 뭐야?"

# 템플릿 완성
prompt = PromptTemplate.from_template(template=template)
print(prompt)

# 3. LLMChain 객체
# 연결된 체인(Chain)객체 생성
llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.invoke({"country": "캐나다"}))
print(llm_chain.invoke({"country": "대한민국"}))

# 4. apply()로 여러 개 입력 처리 한 번에
input_list = [{"country": "호주"}, {"country": "중국"}, {"country": "네덜란드"}]
response = llm_chain.apply(input_list)

print(response[0]["text"])

# input_list 에 대한 결과 반환
result = llm_chain.apply(input_list)

# 반복문으로 결과 출력
for res in result:
    print(res["text"].strip())

# 5. ganerate()
# 입력값
input_list = [{"country": "호주"}, {"country": "중국"}, {"country": "네덜란드"}]

# input_list 에 대한 결과 반환
generated_result = llm_chain.generate(input_list)
print(generated_result)

# 토큰 사용량 출력
print(generated_result.llm_output)

# 6. 2개 이상의 변수를 템플릿 안에 정의
# 질문 템플릿 형식 정의
template = "{area1} 와 {area2} 의 시차는 몇시간이야?"

# 템플릿 완성
prompt = PromptTemplate.from_template(template)
print(prompt)

# 연결된 체인(Chain)객체 생성
llm_chain = LLMChain(prompt=prompt, llm=llm)

# 체인 실행: run()
print(llm_chain.invoke({"area1": "서울", "area2": "파리"}))

input_list = [
    {"area1": "파리", "area2": "뉴욕"},
    {"area1": "서울", "area2": "하와이"},
    {"area1": "켄버라", "area2": "베이징"},
]

# 반복문으로 결과 출력
result = llm_chain.apply(input_list)
for res in result:
    print(res["text"].strip())

# 7. stream으로 실시간 출력
# 객체 생성
llm = ChatOpenAI(
    temperature=0,  # 창의성 (0.0 ~ 2.0)
    max_tokens=2048,  # 최대 토큰수
    model_name="gpt-3.5-turbo",  # 모델명
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

# 질의내용
question = "대한민국에 대해서 300자 내외로 최대한 상세히 알려줘"

# 스트리밍으로 답변 출력
response = llm.invoke(question)
print(response)