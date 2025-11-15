import streamlit as st
import tiktoken
from loguru import logger
import pandas as pd
import re

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
    # 문자열인 경우
    if isinstance(chunk, str):
        return chunk

    # dict인 경우
    elif isinstance(chunk, dict):
        # 여러 가지 키 시도
        for key in ['content', 'text', 'output', 'answer', 'response']:
            if key in chunk and chunk[key]:
                return str(chunk[key])
        return ''

    # content 속성이 있는 경우 (AIMessage, AIMessageChunk 등)
    elif hasattr(chunk, 'content'):
        content = chunk.content
        if isinstance(content, str):
            return content
        elif isinstance(content, list) and len(content) > 0:
            # content가 리스트인 경우 (일부 LLM)
            return str(content[0]) if content[0] else ''
        return str(content) if content else ''

    # text 속성이 있는 경우
    elif hasattr(chunk, 'text'):
        return str(chunk.text) if chunk.text else ''

    # 기타 타입은 문자열로 변환 시도
    else:
        try:
            return str(chunk)
        except:
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
                # 검색 쿼리 향상 (히스토리 컨텍스트 포함)
                search_query = user_input

                # 짧은 질문인 경우 히스토리에서 컨텍스트 추가
                if len(user_input.split()) <= 3 and msgs.messages:
                    # 최근 2개의 사용자 메시지에서 컨텍스트 추출
                    recent_context = []
                    for msg in reversed(msgs.messages[-4:]):  # 최근 4개 메시지 확인
                        if msg.type == "human":
                            recent_context.append(msg.content)
                            if len(recent_context) >= 2:
                                break

                    if recent_context:
                        # 최근 컨텍스트와 현재 질문 결합
                        search_query = " ".join(reversed(recent_context)) + " " + user_input
                        logger.info(f"향상된 검색 쿼리: {search_query}")

                # 검색 실행
                retrieved_docs = retriever.invoke(search_query)
                logger.info(f"원본 질문: {user_input}")
                logger.info(f"검색 쿼리: {search_query}")
                logger.info(f"검색된 문서 수: {len(retrieved_docs)}")
                for i, doc in enumerate(retrieved_docs[:3]):
                    logger.info(f"문서 {i+1}: {doc.page_content[:100]}...")

                # 문서 포맷팅
                context = "\n\n".join(doc.page_content for doc in retrieved_docs)

                # LLM 연결
                llm = RemoteRunnable(DEFAULT_LLM_URL)

                # 프롬프트 구성
                full_prompt = f"""당신은 Future Systems 회사 소개 전문 AI 어시스턴트입니다.

검색된 회사 정보가 질문과 관련이 있다면 그 정보를 우선적으로 사용하여 정확하게 답변하세요.
검색된 정보가 질문과 관련이 없거나 충분하지 않다면 일반 지식을 사용하여 친절하게 답변하세요.
특히 회사 직원의 전화번호, 이메일, 주소 등은 검색된 정보를 정확히 제공하세요.

검색된 회사 정보:
{context}
---

"""

                # 최근 히스토리만 추가 (최근 6개 메시지, 3턴)
                if msgs.messages:
                    recent_messages = msgs.messages[-6:]  # 최근 3턴 (6개 메시지)
                    if recent_messages:
                        full_prompt += "최근 대화:\n"
                        for msg in recent_messages:
                            role = "사용자" if msg.type == "human" else "어시스턴트"
                            full_prompt += f"{role}: {msg.content}\n"
                        full_prompt += "\n"

                # 현재 질문 (명확하게 구분)
                full_prompt += f"현재 질문: {user_input}\n\n답변 (간결하고 정확하게, 프롬프트 형식 없이 내용만):"

                logger.info(f"Full prompt length: {len(full_prompt)}")

                # 응답 생성 (스트리밍)
                with st.chat_message("ai"):
                    chat_container = st.empty()
                    chunks = []

                    # RemoteRunnable에 직접 텍스트 전달
                    for chunk in llm.stream(full_prompt):
                        logger.info(f"Chunk type: {type(chunk)}, content repr: {repr(chunk)}")
                        chunk_text = extract_text_from_chunk(chunk)
                        if chunk_text:
                            chunks.append(chunk_text)
                            chat_container.markdown("".join(chunks))
                        else:
                            logger.warning(f"Empty chunk_text from chunk: {repr(chunk)}")

                    # 응답을 히스토리에 추가
                    final_response = "".join(chunks)
                    logger.info(f"Final response length: {len(final_response)}")

                    # 프롬프트 패턴 제거 (후처리)
                    final_response = final_response.strip()

                    # 모든 프롬프트 패턴 제거 (앞, 뒤, 중간)
                    # "Q: ..." 나 "A: ..." 패턴 제거
                    final_response = re.sub(r'^[QA]:\s*', '', final_response)
                    final_response = re.sub(r'\n[QA]:\s*', '\n', final_response)

                    # "사용자:" "어시스턴트:" 패턴 제거
                    final_response = re.sub(r'^(사용자|어시스턴트):\s*', '', final_response)
                    final_response = re.sub(r'\n(사용자|어시스턴트):\s*', '\n', final_response)

                    # "현재 질문:" 같은 메타 텍스트 제거
                    final_response = re.sub(r'현재 질문:.*?\n', '', final_response)
                    final_response = re.sub(r'답변.*?:', '', final_response, count=1)

                    final_response = final_response.strip()

                    # 정리된 응답 표시
                    if final_response:
                        chat_container.markdown(final_response)
                        msgs.add_user_message(user_input)
                        msgs.add_ai_message(final_response)
                    else:
                        logger.error("No response generated from LLM")
                        error_msg = "LLM에서 응답을 생성하지 못했습니다."
                        chat_container.error(error_msg)

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
