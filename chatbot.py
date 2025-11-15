import streamlit as st
import tiktoken
from loguru import logger
import pandas as pd

from langchain_core.messages import ChatMessage
from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langserve import RemoteRunnable

# 설정
CSV_EMPLOYEE_PATH = "company_data.csv"  # 직원 데이터 (2명)
CSV_COMPANY_INFO_PATH = "company_info_data.csv"  # 회사 소개 데이터 (200개)
PDF_DATA_PATH = "futuresystems_company_brochure.pdf"
DEFAULT_LLM_URL = "https://dioramic-corrin-undetractively.ngrok-free.dev/llm/"

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def load_employee_csv(csv_path):
    """직원 CSV 데이터를 Document 형식으로 로드"""
    df = pd.read_csv(csv_path)
    documents = []

    for idx, row in df.iterrows():
        content = f"""이름: {row['이름']}
직급: {row['직급']}
부서: {row['부서']}
전화번호: {row['전화번호']}
이메일: {row['이메일']}
입사일: {row['입사일']}
담당업무: {row['담당업무']}"""

        doc = Document(
            page_content=content,
            metadata={
                "source": "company_data.csv",
                "name": row['이름'],
                "position": row['직급'],
                "phone": row['전화번호']
            }
        )
        documents.append(doc)

    return documents

def load_company_info_csv(csv_path):
    """회사 소개 CSV 데이터를 Document 형식으로 로드"""
    df = pd.read_csv(csv_path)
    documents = []

    for idx, row in df.iterrows():
        content = f"""[{row['카테고리']}] {row['항목']}
{row['내용']}"""

        doc = Document(
            page_content=content,
            metadata={
                "source": "company_info_data.csv",
                "category": row['카테고리'],
                "item": row['항목']
            }
        )
        documents.append(doc)

    return documents

def load_pdf_data(pdf_path):
    """PDF 데이터 로드"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def extract_text_from_chunk(chunk):
    """응답 텍스트 추출"""
    if isinstance(chunk, str):
        return chunk
    elif isinstance(chunk, dict):
        content = chunk.get('content', '') or chunk.get('text', '')
        return content if content else ''
    elif hasattr(chunk, 'content'):
        return chunk.content if chunk.content else ''
    else:
        return ''

@st.cache_resource
def initialize_rag_system():
    """RAG 시스템 초기화 (자동)"""
    all_docs = []

    # 직원 CSV 로드
    try:
        employee_docs = load_employee_csv(CSV_EMPLOYEE_PATH)
        all_docs.extend(employee_docs)
    except Exception as e:
        logger.error(f"직원 CSV 로드 실패: {e}")

    # 회사 소개 CSV 로드 (200개)
    try:
        company_info_docs = load_company_info_csv(CSV_COMPANY_INFO_PATH)
        all_docs.extend(company_info_docs)
    except Exception as e:
        logger.error(f"회사 소개 CSV 로드 실패: {e}")

    # PDF 로드
    try:
        pdf_docs = load_pdf_data(PDF_DATA_PATH)
        pdf_chunks = get_text_chunks(pdf_docs)
        all_docs.extend(pdf_chunks)
    except Exception as e:
        logger.error(f"PDF 로드 실패: {e}")

    # 벡터스토어 생성
    if all_docs:
        vectorstore = get_vectorstore(all_docs)
        retriever = vectorstore.as_retriever(
            search_type='mmr',
            search_kwargs={'k': 10, 'fetch_k': 20}
        )
        return retriever
    else:
        logger.error("로드된 문서가 없습니다.")
        return None

def main():
    st.set_page_config(
        page_title="챗봇",
        page_icon=""
    )

    st.title(":blue[챗봇]")

    # StreamlitChatMessageHistory 초기화 (공식 LangChain 방식)
    msgs = StreamlitChatMessageHistory(key="chat_messages")

    # RAG 시스템 초기화 (자동, 캐싱)
    retriever = initialize_rag_system()

    # 대화 기록 출력
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # 사용자 입력
    if user_input := st.chat_input("질문을 입력하세요"):
        # 사용자 메시지 표시
        st.chat_message("human").write(user_input)

        if retriever:
            try:
                # 검색 결과 확인 (디버깅)
                retrieved_docs = retriever.invoke(user_input)
                logger.info(f"검색 질문: {user_input}")
                logger.info(f"검색된 문서 수: {len(retrieved_docs)}")
                for i, doc in enumerate(retrieved_docs[:3]):
                    logger.info(f"문서 {i+1}: {doc.page_content[:100]}...")

                # 문서 포맷팅
                context = "\n\n".join(doc.page_content for doc in retrieved_docs)

                # LLM 연결
                llm = RemoteRunnable(DEFAULT_LLM_URL)

                # 프롬프트에 컨텍스트와 히스토리 주입
                messages = [
                    ("system", f"""다음 제공된 정보만을 사용하여 질문에 답변하세요.
제공된 정보에 없는 내용은 절대 답변하지 마세요.
전화번호, 주소, 이메일 등은 제공된 정보에서 정확히 찾아서 그대로 제공하세요.

제공된 정보:
{context}""")
                ]

                # 히스토리 추가
                messages.extend([(msg.type, msg.content) for msg in msgs.messages])

                # 현재 질문 추가
                messages.append(("human", user_input))

                # 프롬프트 생성
                chat_prompt = ChatPromptTemplate.from_messages(messages)

                # 체인 실행
                chain = chat_prompt | llm

                # 응답 생성 (스트리밍)
                with st.chat_message("ai"):
                    chat_container = st.empty()
                    chunks = []

                    for chunk in chain.stream({}):
                        chunk_text = extract_text_from_chunk(chunk)
                        if chunk_text:
                            chunks.append(chunk_text)
                            chat_container.markdown("".join(chunks))

                    # 응답을 히스토리에 추가
                    final_response = "".join(chunks)
                    if final_response:
                        msgs.add_user_message(user_input)
                        msgs.add_ai_message(final_response)

            except Exception as e:
                error_msg = f"오류가 발생했습니다: {str(e)}"
                logger.error(error_msg)
                logger.error(f"에러 상세: {e}")
                st.chat_message("ai").error(error_msg)
        else:
            error_msg = "RAG 시스템이 초기화되지 않았습니다."
            st.chat_message("ai").error(error_msg)

if __name__ == '__main__':
    main()
