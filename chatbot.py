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
from langchain_core.runnables import RunnablePassthrough
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
            search_kwargs={'k': 5, 'fetch_k': 10}
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

    # 메시지 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # RAG 시스템 초기화 (자동, 캐싱)
    retriever = initialize_rag_system()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 대화 기록 출력
    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)

    RAG_PROMPT_TEMPLATE = """아래 정보를 바탕으로 질문에 답변하세요.
전화번호, 이메일 등의 정보는 정확히 제공하세요.

질문: {question}
정보: {context}
답변:"""

    # 사용자 입력
    if user_input := st.chat_input("질문을 입력하세요"):
        # 사용자 메시지 추가
        st.session_state.messages.append(ChatMessage(role="user", content=user_input))
        st.chat_message("user").write(user_input)

        # AI 응답
        with st.chat_message("assistant"):
            chat_container = st.empty()

            if retriever:
                try:
                    # LLM 연결
                    llm = RemoteRunnable(DEFAULT_LLM_URL)

                    # 프롬프트 생성
                    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

                    # 문서 포맷팅
                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)

                    # RAG 체인
                    rag_chain = (
                        {
                            "context": retriever | format_docs,
                            "question": RunnablePassthrough(),
                        }
                        | prompt
                        | llm
                    )

                    # 스트리밍 응답
                    answer = rag_chain.stream(user_input)
                    chunks = []
                    for chunk in answer:
                        chunk_text = extract_text_from_chunk(chunk)
                        if chunk_text:
                            chunks.append(chunk_text)
                            chat_container.markdown("".join(chunks))

                    # 대화 기록에 추가
                    st.session_state.messages.append(
                        ChatMessage(role="assistant", content="".join(chunks))
                    )

                except Exception as e:
                    error_msg = f"오류가 발생했습니다: {str(e)}"
                    chat_container.error(error_msg)
                    st.session_state.messages.append(
                        ChatMessage(role="assistant", content=error_msg)
                    )
            else:
                error_msg = "RAG 시스템이 초기화되지 않았습니다."
                chat_container.error(error_msg)
                st.session_state.messages.append(
                    ChatMessage(role="assistant", content=error_msg)
                )

if __name__ == '__main__':
    main()
