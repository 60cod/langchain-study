from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. 기본 예시: PromptTemplate + Model(LLM) + StrOutputParser
# template 정의
template = "{country}의 수도는 어디인가요?"

# from_template 메소드를 이용하여 PromptTemplate 객체 생성
prompt_template = PromptTemplate.from_template(template)
print(prompt_template)

# prompt 생성
prompt = prompt_template.format(country="대한민국")
print(prompt)

prompt = prompt_template.format(country="미국")
print(prompt)

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    max_tokens=2048,
    temperature=0.1,
)

output = StrOutputParser()

# 2. Chain 생성
# 주어진 나라에 대하여 수도를 묻는 프롬프트 템플릿을 생성합니다.
# prompt = PromptTemplate.from_template("{country}의 수도는 어디인가요?")

# 주어진 나라에 대하여 수도를 묻는 프롬프트 템플릿을 생성합니다.
template = """
당신은 친절하게 답변해 주는 친절 봇입니다. 사용자의 질문에 [FORMAT]에 맞추어 답변해 주세요.
답변은 항상 한글로 작성해 주세요.

질문:
{question}에 대하여 설명해 주세요.

FORMAT:
- 개요:
- 예시:
- 출처:
"""

template = """
당신은 영어를 가르치는 10년차 영어 선생님입니다. 상황에 [FORMAT]에 영어 회화를 작성해 주세요.

상황:
{question}

FORMAT:
- 영어 회화:
- 한글 해석:
"""


prompt = PromptTemplate.from_template(template)


# OpenAI 챗모델을 초기화합니다.
model = ChatOpenAI(model="gpt-4-turbo-preview")
# 문자열 출력 파서를 초기화합니다.
output_parser = StrOutputParser()

# 프롬프트, 모델, 출력 파서를 연결하여 처리 체인을 구성합니다.
chain = prompt | model | output_parser

# 완성된 Chain 을 이용하여 country 를 '대한민국'으로 설정하여 실행합니다.
# chain.invoke({"country": "대한민국"})
print(chain.invoke({"question": "저는 식당에 가서 음식을 주문하고 싶어요"}))

# 완성된 Chain 을 이용하여 question 를 '미국에서 피자 주문'으로 설정하여 실행합니다.
print(chain.invoke({"question": "미국에서 피자 주문"}))

# 3. 전체 파이프라인
# prompt 를 PromptTemplate 객체로 생성합니다.
prompt = PromptTemplate.from_template("{topic} 에 대해 쉽게 설명해주세요.")

# input 딕셔너리에 주제를 'ice cream'으로 설정합니다.
input = {"topic": "양자역학"}

# prompt 객체의 invoke 메서드를 사용하여 input을 전달하고 대화형 프롬프트 값을 생성합니다.
prompt.invoke(input)

# prompt 객체와 model 객체를 파이프(|) 연산자로 연결하고 invoke 메서드를 사용하여 input을 전달합니다.
# 이를 통해 AI 모델이 생성한 메시지를 반환합니다.
print((prompt | model).invoke(input))

# parse_output 메서드를 사용하여 AI 모델이 생성한 메시지 문자열로 출력합니다.
print((prompt | model | output_parser).invoke(input))