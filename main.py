import openai  # OpenAI API 호출
import os  # 경로 및 환경 변수 관리

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import GoogleSearchAPIWrapper

import streamlit as st
from dotenv import load_dotenv

# 환경 변수 불러오기
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Google 검색 API 설정
search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)

# 파인튜닝한 모델
em_model = "ft:gpt-3.5-turbo-1106:personal:kimstitute-hsu-mdai2:AeZ66shC"

# 랭체인 LLM 설정
llm = ChatOpenAI(model=em_model, openai_api_key=OPENAI_API_KEY)  # 파인튜닝된 모델을 호출

# 프롬프트 템플릿 정의
PROMPT_FOR_LLM_RAG = """
Refer to the content of the 'Search Results Summary' as much as possible when providing your answer.
Question: {query}

Search Results Summary:
{context}

You are an medical expert. 
Based on the summarized search results, provide critical medical advice grounded in reliable knowledge to the patient \
or their guardian in an medical situation. 
Please, avoid unnecessary lengthy explanations. All responses must be in Korean.
If the question is completely unrelated to a medicine or pharmacy, respond with: 
'저는 의료 지식이 필요한 상황에서만 답변할 수 있습니다.'
"""


prompt = PromptTemplate(input_variables=["query", "context"], template=PROMPT_FOR_LLM_RAG)


# 구글링한 정보를 주고 모델 응답 가져오기
def googling_response(query):
    # Google Search 실행
    search_results = search.run(query)

    # 검색 결과가 문자열로 반환된 경우 처리
    if isinstance(search_results, str):
        if search_results.strip() == "No good Google Search Result was found":
            return "검색 결과를 찾을 수 없습니다."
        search_results = search_results.splitlines()  # 줄 단위로 리스트 변환

    # 검색 결과가 리스트 형태일 경우 처리
    if isinstance(search_results, list) and search_results:
        context = "\n".join(search_results[:5])  # 상위 5개의 검색 결과 요약
        prompt_text = prompt.format(query=query, context=context)
        return llm.predict(prompt_text)
    else:
        return "검색 결과를 찾을 수 없습니다."


# 구글링한 정보만 반환하는 함수
def googling_context(query, k):
    search_results = search.run(query)
    if isinstance(search_results, str):
        if search_results.strip() == "No good Google Search Result was found":
            return "검색 결과를 찾을 수 없습니다."
        search_results = search_results.splitlines()

    if isinstance(search_results, list) and search_results:
        # 상위 k개의 요약 결과를 정리
        clean_results = [result.strip() for result in search_results[:k] if result.strip()]
        return "\n".join(clean_results)
    else:
        return "검색 결과를 찾을 수 없습니다."


PROMPT_FOR_LLM_UNLIMITED = "you are friendly adviser"

PROMPT_FOR_LLM_DOCTOR = \
    "You are an emergency medical expert who has received contact from a patient or their guardian \
    in a critical situation. Please provide brief and essential medical advice based on reliable information. \
    , and avoid unnecessary lengthy explanations due to the urgency of the situation. \
    If the question is completely unrelated to a medical situation, respond with: \
    '저는 응급 의료 전문 AI로써 의료 상황에만 답변할 수 있습니다.'"

PROMPT_FOR_LLM_COT = \
    "Use a Chain of Thought approach to answer the patient's question. You must explain your reasoning step by step, \
    clearly stating the rationale for each step. Each step must logically flow into the next, ensuring \
    a well-reasoned and coherent conclusion. Number each step to create a structured and easy-to-follow response. \
    Each step must include a justification that starts with '이유:'. Have to add a line break after each step \
    to improve readability. Every step must contribute directly to building a comprehensive and clear answer."


PROMPT_FOR_LLM = PROMPT_FOR_LLM_DOCTOR


# API 호출하여 파인튜닝된 모델에 응답 요청하는 함수
def get_gpt_response(prompt):
    try:
        response = openai.chat.completions.create(
            model=em_model,
            messages=[
                {"role": "system", "content": PROMPT_FOR_LLM},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Error:", e)
        return None


# ReAct 방법론을 사용해서 응답을 출력하는 함수
def get_react_response(subject):  # 매개변수는 사용자 입력 프롬프트와
    try:
        global PROMPT_FOR_LLM
        PROMPT_FOR_LLM = PROMPT_FOR_LLM_UNLIMITED

        reasoning = get_gpt_response(f"How to solve {subject}? \
        Provide your reasoning in only one sentence. response must be in Korean.\
        Start your response with a complete sentence in Korean.")

        action = get_gpt_response(f"What search query would be helpful to address '{reasoning}'? \
        All responses must be in Korean. and search query must be few words\
        Provide the search query as a single complete sentence in Korean.")
        observation = googling_context(action, 3)

        final_answer = get_gpt_response(
            f"How should I answer about {subject}? Use {reasoning} and {observation} to provide a response. \
            Provide your reasoning in one or two sentence. All responses must be in Korean. \
            starting with a complete sentence.")

        # HTML 포맷팅
        response = f"""
                <div>
                    <p><strong style="color:#3498db;">사고 과정 (Reasoning):</strong> {reasoning}</p>
                    <p><strong style="color:#e74c3c;">행동 (Action):</strong> {action}</p>
                    <p><strong style="color:#f39c12;">관찰 (Observation):</strong> {observation}</p>
                    <p><strong style="color:#27ae60;">최종 응답 (Final Answer):</strong> {final_answer}</p>
                </div>
                """

        return response
    except Exception as e:
        return f"ReAct 처리 중 오류 발생: {str(e)}"


def main():
    # HTML과 CSS로 UI 디자인
    st.markdown("""
        <style>
            .stTextArea textarea {
                background-color: white;
                font-size: 16px;
                line-height: 1.5;
            }
            .solution-box {
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                font-size: 16px;
                line-height: 1.5;
                color: black;
                max-height: 300px; /* 최대 높이 설정 */
                overflow-y: auto; /* 스크롤 추가 */
            }
        </style>
        """, unsafe_allow_html=True)

    st.markdown(
        "<img src='https://raw.githubusercontent.com/kimstitute/Medical_AI/main/sangsang2.png' \
        style='height: 60px; margin-right: 10px; margin-bottom: 18px;' alt='bugi'>"
        "<span style='font-size: 40px; font-weight: bold;'>인공지능 응급 의료 챗봇</span> "
        "<span style='font-size: 10px; color: gray;'>Created by HSU AI/CS KIM MINSANG</span>",
        unsafe_allow_html=True
    )

    # 사용자가 질문(프롬프트)을 입력하는 입력창
    subject = st.text_area("환자의 정보와 응급 상황을 간단하게 설명해주세요.", height=150)

    # 체크박스들
    global PROMPT_FOR_LLM
    global PROMPT_FOR_LLM_DOCTOR
    global PROMPT_FOR_LLM_COT
    use_cot = st.checkbox("CoT(AI 사고 과정 보기) 사용")
    use_rag = st.checkbox("RAG(검색 증강 생성) 사용")
    use_react = st.checkbox("ReAct(사고 + 검색) 사용")

    if st.button("솔루션 작성"):
        with st.spinner("솔루션 작성 중..."):
            result = None  # 초기화

            # 1. 기본 응답 생성
            if not use_cot and not use_rag and not use_react:
                PROMPT_FOR_LLM = PROMPT_FOR_LLM_DOCTOR
                result = get_gpt_response(subject)

            # 3. RAG
            elif not use_cot and use_rag and not use_react:
                PROMPT_FOR_LLM = PROMPT_FOR_LLM_DOCTOR
                rag_result = googling_response(subject)
                if rag_result:  # 검색 결과가 있는 경우 RAG 사용
                    result = rag_result
                else:  # 검색된 문서가 없을 경우 기본 모델 응답 제공
                    st.warning("검색된 문서가 없습니다. 기본 모델로 답변을 생성합니다.")
                    result = get_gpt_response(subject)

            # 5. CoT + RAG
            elif use_cot and use_rag and not use_react:
                PROMPT_FOR_LLM = PROMPT_FOR_LLM_COT + PROMPT_FOR_LLM_DOCTOR
                search_context = googling_context(subject, 5)
                if search_context: # 검색 결과가 있는 경우 RAG 사용
                    final_subject = f"Context:\n{search_context}\n\nQuestion:\n{subject}"
                    result = get_gpt_response(final_subject)
                else:  # 검색된 문서가 없을 경우 CoT 응답 제공
                    st.warning("검색된 문서가 없습니다. CoT로만 답변을 생성합니다.")
                    result = get_gpt_response(subject)

            # 6. ReAct, ReAct + RAG
            elif not use_cot and use_react:
                result = get_react_response(subject)

            # COT, CoT+ReAct, CoT+RAG+ReAct는 COT 단독으로 취급
            else:
                PROMPT_FOR_LLM = PROMPT_FOR_LLM_COT + PROMPT_FOR_LLM_DOCTOR
                result = get_gpt_response(subject)

            if result:
                st.markdown("### 생성된 솔루션")
                st.markdown(f"<div class='solution-box'>{result}</div>", unsafe_allow_html=True)
            else:
                st.error("솔루션 생성 실패")

            st.write("※본 솔루션은 인공지능에 의해 생성된 참고 정보이며, 실제 의료 판단이나 진단을 대신할 수 없습니다. \
            정확한 진료 및 치료를 위해 반드시 의료 전문가의 상담을 받으시기 바랍니다.")


if __name__ == "__main__":
    main()
