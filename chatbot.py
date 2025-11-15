import streamlit as st
import tiktoken
from loguru import logger
import pandas as pd

from langchain_core.messages import ChatMessage
from langchain.schema import Document

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langserve import RemoteRunnable

# 설정
CSV_DATA_PATH = "company_data.csv"
PDF_DATA_PATH = "futuresystems_company_brochure.pdf"
DEFAULT_LLM_URL = "https://dioramic-corrin-undetractively.ngrok-free.dev/llm/"

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def load_csv_data(csv_path):
    """CSV 데이터를 Document 형식으로 로드"""
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
    with st.spinner("데이터 로딩 중..."):
        all_docs = []

        # CSV 로드
        try:
            csv_docs = load_csv_data(CSV_DATA_PATH)
            all_docs.extend(csv_docs)
            st.success(f"✅ 직원 데이터 로드: {len(csv_docs)}명")
        except Exception as e:
            st.warning(f"CSV 로드 실패: {e}")

        # PDF 로드
        try:
            pdf_docs = load_pdf_data(PDF_DATA_PATH)
            pdf_chunks = get_text_chunks(pdf_docs)
            all_docs.extend(pdf_chunks)
            st.success(f"✅ 회사 소개 자료 로드: {len(pdf_chunks)}개 청크")
        except Exception as e:
            st.warning(f"PDF 로드 실패: {e}")

        # 벡터스토어 생성
        if all_docs:
            vectorstore = get_vectorstore(all_docs)
            retriever = vectorstore.as_retriever(
                search_type='mmr',
                search_kwargs={'k': 5, 'fetch_k': 10}
            )
            st.success(f"✅ RAG 시스템 준비 완료: 총 {len(all_docs)}개 문서")
            return retriever
        else:
            st.error("로드된 문서가 없습니다.")
            return None

def main():
    st.set_page_config(
        page_title="챗봇",
        page_icon=""
    )

    st.title(":blue[챗봇]")
    st.caption("🚀 RAG 하이브리드 챗봇 - 자동 로드")

    # 메시지 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 사이드바
    with st.sidebar:
        st.header("설정")

        llm_url = st.text_input(
            "LLM 서버 URL",
            value=DEFAULT_LLM_URL,
            help="ngrok URL이 변경되면 여기에 입력하세요"
        )

        st.divider()
        st.info("""
        **자동 로드됨:**
        - 직원 데이터 (CSV)
        - 회사 소개 (PDF)

        바로 질문하세요!
        """)

    # RAG 시스템 초기화 (자동, 캐싱)
    retriever = initialize_rag_system()

    if "messages" not in st.session_state or len(st.session_state["messages"]) == 0:
        st.session_state["messages"] = [
            ChatMessage(role="assistant", content="안녕하세요! 회사 및 직원 정보에 대해 궁금하신 점을 물어보세요!")
        ]

    # 대화 기록 출력
    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)

    RAG_PROMPT_TEMPLATE = """당신은 Future Systems 회사 소개 전문 AI 어시스턴트입니다.
검색된 문서 내용을 바탕으로 회사 및 직원 정보에 대한 질문에 친절하고 정확하게 답변해주세요.
특히 전화번호, 이메일, 부서 등의 정보는 검색된 내용을 정확히 그대로 제공하세요.

Question: {question}
Context: {context}
Answer:"""

    # 사용자 입력
    if user_input := st.chat_input("궁금한 점을 물어보세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append(ChatMessage(role="user", content=user_input))
        st.chat_message("user").write(user_input)

        # AI 응답
        with st.chat_message("assistant"):
            chat_container = st.empty()

            if retriever:
                try:
                    # LLM 연결
                    llm = RemoteRunnable(llm_url)

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
