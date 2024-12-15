import openai  # OpenAI API 호출
import os  # 경로 및 환경 변수 관리
from langchain_community.vectorstores import Pinecone as LangchainPinecone  # Pinecone과의 통합을 위해 사용
from langchain_openai.embeddings import OpenAIEmbeddings  # 벡터 임베딩 생성
from langchain_openai import ChatOpenAI
from pinecone import Pinecone, ServerlessSpec  # Pinecone 벡터 데이터베이스 사용
from langchain.chains import RetrievalQA  # RAG(검색 증강 생성)를 구현하기 위해 사용
import streamlit as st
from dotenv import load_dotenv

# 환경 변수 불러오기
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Pinecone 초기화 및 설정
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-ai"  # Pinecone에서 쓰는 DB 이름

# 인덱스 불러오기
if index_name not in pc.list_indexes().names(): # 인덱스가 Pinecone 서버에 존재 않을 경우 새로 생성
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI text-embedding-ada-002의 차원
        metric="cosine",  # 벡터 간 유사도 측정할 때 코사인 유사도 사용
        spec=ServerlessSpec(cloud="aws", region="us-west-1")  # Pinecone의 무서버 환경 설정
    )

# Pinecone 인덱스 연결
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)  # OpenAI API로 텍스트 데이터 벡터로 임베딩
vector_store = LangchainPinecone.from_existing_index(index_name=index_name, embedding=embeddings)  # 검색 위해 연결

# 파인튜닝한 모델
em_model = "ft:gpt-3.5-turbo-1106:personal:kimstitute-hsu-mdai2:AeZ66shC"

# LangChain QA 체인 설정
llm = ChatOpenAI(model=em_model, openai_api_key=OPENAI_API_KEY)  # 파인튜닝된 모델을 호출
qa_chain = RetrievalQA.from_chain_type(  # LangChain 통해 RAG(검색 증강 생성)를 구성
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()  # Pinecone을 검색 도구로 사용하도록 설정
)


PROMPT_FOR_LLM_DOCTOR = \
    "You are an emergency medical expert who has received contact from a patient or their guardian \
    in a critical situation. Please provide brief and essential medical advice based on reliable information. \
    , and avoid unnecessary lengthy explanations due to the urgency of the situation. \
    if the input is irrelevant to a medical emergency, respond with: '저는 응급 의료 전문 AI로써 의료 상황에만 답변할 수 있습니다.'"

PROMPT_FOR_LLM_COT = \
    "Use a Chain of Thought approach to answer the patient's question. You must explain your reasoning step by step, \
    clearly stating the rationale for each step. Each step must logically flow into the next, ensuring \
    a well-reasoned and coherent conclusion. Number each step to create a structured and easy-to-follow response. \
    Each step must include a justification that starts with '이유:'. Add a line break after each step \
    to improve readability. Every step must contribute directly to building a comprehensive and clear answer."

PROMPT_FOR_REACT = """
You are a highly capable reasoning and acting agent. Answer questions by reasoning step-by-step.
When needed, perform actions such as searching, querying a database, or retrieving external data.
Each response must strictly adhere to the following format and be written in Korean:
Format:
사고 과정(Reasoning): [Your reasoning here]
행동(Action): [Please ensure to include the keywords you want to search inside '[]', separated by commas]
관찰(Observation): [Result of the action]
최종 응답(Final Answer): [Your final answer based on the reasoning and observations]
"""


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
def get_react_response(prompt, use_rag=False):  # 매개변수는 사용자 입력 프롬프트와 RAG 사용 여부
    reasoning = []  # ReAct 동작 중 사고 과정 저장하는 리스트
    try:
        # Reasoning 시작, RAG 가능하면, RAG로 아니면, 그냥 모델 응답 사용
        llm_response = qa_chain.invoke({"query": prompt}) if use_rag else get_gpt_response(prompt)

        # LLM 응답서 Reasoning/Action/Observation 추출
        lines = llm_response.get("result", "").splitlines()  # LLM 응답 줄별로 분리해서 리스트에 넣음
        final_answer = ""  # 최종 답변 저장하는 빈 문자열
        for line in lines:
            if line.startswith("사고"):  # 사고로 문장 시작하면 사고 과정 정보로 간주
                reasoning.append(line[len("Reasoning:"):].strip())  # 사고 과정 저장하는 리스트에 넣기
            elif line.startswith("행동(Action): ["):  # 행동 단계에서 검색 수행했던 작업들 추출
                query = line[len("행동(Action): search("):-1].strip()
                search_results = vector_store.similarity_search(query, k=5)  # Pinecone 벡터 DB에서 검색
                if search_results:
                    observation = "\n".join([doc.page_content for doc in search_results])  # 검색된 문서의 내용을 줄 단위로 연결
                    reasoning.append(f"Observation: {observation}")  # 사고 과정 리스트에 추가
                else:
                    reasoning.append("Observation: No relevant documents found.")
            elif line.startswith("최종 응답"):  # 최종 응답으로 시작하면 최종 응답으로 간주
                final_answer = line[len("Final Answer:"):].strip()  # 텍스트를 잘라내고 좌우 공백을 제거

        return "\n".join(reasoning) + f"\n\n{final_answer}"  # reasoning 리스트에 저장된 모든 요소를 연결하고 최종 응답 추가
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

    # 사용자가 질문을 입력하는 입력창
    subject = st.text_area("환자의 정보와 응급 상황을 간단하게 설명해주세요.", height=150)

    # 체크박스들
    global PROMPT_FOR_LLM
    use_cot = st.checkbox("CoT(AI 사고 과정 보기) 사용")
    use_rag = st.checkbox("RAG(검색 증강 생성) 사용")
    use_react = st.checkbox("ReAct(사고 + 검색) 사용")

    if st.button("솔루션 작성"):
        with st.spinner("솔루션 작성 중..."):
            result = None  # 초기화
            # 버튼 클릭 시 체크박스 상태 반영
            PROMPT_FOR_LLM = PROMPT_FOR_LLM_COT + PROMPT_FOR_LLM_DOCTOR if use_cot else PROMPT_FOR_LLM_DOCTOR
            if use_react:
                PROMPT_FOR_LLM = PROMPT_FOR_REACT
                result = get_react_response(subject, use_rag)
            else:
                if use_rag:
                    # Pinecone 벡터 DB서 사용자 입력과 가장 유사한 문서 최대 5개까지 검색
                    search_results = vector_store.similarity_search(subject, k=5)
                    if search_results:
                        # 검색 결과가 있는 경우 RAG 사용
                        search_context = "\n".join([doc.page_content for doc in search_results])
                        if use_cot:
                            final_subject = f"Context:\n{search_context}\n\nQuestion:\n{subject}\n\nRole:\n{PROMPT_FOR_LLM_COT}"
                        else:
                            final_subject = f"Context:\n{search_context}\n\nQuestion:\n{subject}"
                        try:
                            response = qa_chain.invoke({"query": final_subject})  # LangChain QA 체인을 통해 최종 프롬프트 전달, 응답 생성
                            result = response.get("result", "검색된 문서가 없습니다.")  # 응답 존재 시 result 키, 응답 없으면 기본 메시지
                        except Exception as e:
                            result = f"RAG 처리 중 오류 발생: {str(e)}"
                    else:
                        # 검색된 문서가 없을 경우 기본 모델 응답 제공
                        st.warning("검색된 문서가 없습니다. 기본 모델로 답변을 생성합니다.")
                        result = get_gpt_response(subject)
                else:
                    # RAG를 사용하지 않는 경우 기본 모델 응답 제공
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
