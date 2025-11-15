import streamlit as st
import tiktoken
from loguru import logger
import pandas as pd
import re

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
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

def clean_response_text(response):
    """LLM 응답에서 프롬프트 패턴 제거"""
    response = response.strip()

    # 프롬프트 형식 패턴 제거
    patterns_to_clean = [
        (r'^[QA]:\s*', ''),
        (r'\n[QA]:\s*', '\n'),
        (r'^(사용자|어시스턴트|AI):\s*', ''),
        (r'\n(사용자|어시스턴트|AI):\s*', '\n'),
        (r'^답변\s*[:：]\s*', ''),
        (r'현재.*?질문\s*[:：].*?\n', ''),
    ]

    for pattern, replacement in patterns_to_clean:
        response = re.sub(pattern, replacement, response, flags=re.MULTILINE)

    response = response.strip()

    # 중복 공백 정리
    response = re.sub(r'\n\s*\n', '\n\n', response)
    response = re.sub(r'\n{3,}', '\n\n', response)

    return response

def extract_recent_context(messages, max_messages=4):
    """최근 사용자 메시지에서 컨텍스트 추출"""
    recent_context = []
    for msg in reversed(messages[-max_messages:]):
        if msg.type == "human":
            recent_context.append(msg.content)
            if len(recent_context) >= 2:
                break
    return list(reversed(recent_context))

def extract_most_recent_entity(messages):
    """가장 최근에 언급된 엔티티(인물, 회사명 등) 추출"""
    # 최근 2턴(4개 메시지)만 확인
    recent_msgs = messages[-4:] if len(messages) >= 4 else messages

    # 일반적인 엔티티 패턴 (한국어 이름, 회사명 등)
    entities = []

    for msg in reversed(recent_msgs):
        if msg.type == "human":
            # 한글 이름 패턴 (2-4자)
            names = re.findall(r'[가-힣]{2,4}(?=의|이|은|는|를|을|에게|한테)', msg.content)
            entities.extend(names)

            # 특정 키워드 체크
            if "퓨쳐시스템" in msg.content or "Future Systems" in msg.content:
                entities.append("퓨쳐시스템")

    # 가장 최근 엔티티 반환
    return entities[0] if entities else None

def rewrite_followup_query(user_input, recent_messages):
    """Follow-up 질문을 standalone 질문으로 재작성"""
    # 가장 최근 엔티티 추출
    entity = extract_most_recent_entity(recent_messages)

    if not entity:
        return user_input

    # "전화번호는?", "이메일은?", "직급은?" 같은 패턴
    simple_patterns = ["전화번호", "이메일", "직급", "부서", "담당업무", "주소", "연락처"]

    for pattern in simple_patterns:
        if pattern in user_input and len(user_input) <= 15:
            # "전화번호는?" → "염재준의 전화번호는?"
            return f"{entity}의 {user_input}"

    return user_input

def is_meta_or_followup_question(user_input, has_history):
    """메타 질문 또는 follow-up 질문 감지"""
    # 대화 참조 키워드
    reference_keywords = ["이전", "그", "그사람", "방금", "아까", "위", "앞서"]

    # 언어/변환 키워드
    language_keywords = ["영어", "일본어", "중국어", "한국어", "번역", "translate", "요약", "summarize", "변경"]

    # 메타 질문: 대화 참조 + 언어/변환 키워드
    has_reference = any(keyword in user_input for keyword in reference_keywords)
    has_language = any(keyword in user_input for keyword in language_keywords)

    is_meta = has_reference or (has_language and has_history)
    is_followup = len(user_input.split()) <= 3 and has_history and not has_language

    return is_meta, is_followup

def build_prompt(context, is_meta_question, recent_messages, user_input):
    """프롬프트 구성"""
    if is_meta_question:
        # 메타 질문: 이전 대화 자체에 대한 작업 (번역, 요약 등)
        prompt = """SYSTEM: You are a helpful AI assistant specialized in conversation operations like translation, summarization, and reformatting.

INSTRUCTION: The user wants you to perform an operation on their previous conversation history. You must complete the user's request by processing ALL messages in the conversation history.

EXAMPLES:
Example 1 - Translation to English:
Conversation History:
User: 일본의 수도는?
Assistant: 도쿄입니다.

User Request: 이전 대화를 영어로 변경해줘

Response:
User: What is the capital of Japan?
Assistant: It is Tokyo.

Example 2 - Translation to Japanese:
Conversation History:
User: 정원규 직급은?
Assistant: 대표이사입니다.

User Request: 이전 대화를 일본어로

Response:
ユーザー: チョン・ウォンギュの役職は何ですか？
アシスタント: 代表理事です。

---

NOW PERFORM THE SAME TASK:

"""
        # 히스토리를 명확하게 표시
        if recent_messages:
            prompt += "Conversation History:\n"
            for msg in recent_messages:
                role = "User" if msg.type == "human" else "Assistant"
                prompt += f"{role}: {msg.content}\n"
            prompt += "\n"

        prompt += f"User Request: {user_input}\n\n"
        prompt += "Response:"

    else:
        # 일반 질문: 검색 정보 + 히스토리 균형
        prompt = f"""당신은 Future Systems 회사 소개 전문 AI 어시스턴트입니다.

검색된 회사 정보가 질문과 관련이 있다면 그 정보를 우선적으로 사용하여 정확하게 답변하세요.
검색된 정보가 질문과 관련이 없거나 충분하지 않다면 일반 지식을 사용하여 친절하게 답변하세요.
특히 회사 직원의 전화번호, 이메일, 주소 등은 검색된 정보를 정확히 제공하세요.

검색된 회사 정보:
{context}
---

"""
        # 히스토리 추가 (일반 질문) - 최근 2턴만
        if recent_messages:
            # 최근 2턴(4개 메시지)만 사용
            recent_short = recent_messages[-4:] if len(recent_messages) > 4 else recent_messages

            if recent_short:
                prompt += "최근 대화 (가장 최근 주제 우선):\n"
                for msg in recent_short:
                    role = "사용자" if msg.type == "human" else "AI"
                    prompt += f"{role}: {msg.content}\n"

                # 가장 최근 주제 강조
                last_human_msg = None
                for msg in reversed(recent_short):
                    if msg.type == "human":
                        last_human_msg = msg.content
                        break

                if last_human_msg:
                    prompt += f"\n중요: 현재 대화는 '{last_human_msg}'에 대한 것입니다.\n"
                prompt += "\n"

        # 현재 질문
        prompt += f"현재 질문: {user_input}\n\n답변:"

    return prompt

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
                # 메타/follow-up 질문 감지
                is_meta_question, is_followup = is_meta_or_followup_question(
                    user_input, bool(msgs.messages)
                )

                # 메타 질문이 아닐 때만 검색 실행
                context = ""
                if not is_meta_question:
                    # Query Rewriting: follow-up 질문을 standalone으로 재작성
                    search_query = user_input
                    if is_followup and msgs.messages:
                        rewritten_query = rewrite_followup_query(user_input, msgs.messages)
                        if rewritten_query != user_input:
                            search_query = rewritten_query
                            logger.info(f"Query Rewriting: '{user_input}' → '{search_query}'")
                        else:
                            # 재작성 실패 시 기존 방식 사용
                            recent_context = extract_recent_context(msgs.messages)
                            if recent_context:
                                search_query = " ".join(recent_context) + " " + user_input
                                logger.info(f"컨텍스트 추가: {search_query}")

                    # 검색 실행
                    retrieved_docs = retriever.invoke(search_query)
                    logger.info(f"원본 질문: {user_input}")
                    logger.info(f"검색 쿼리: {search_query}")
                    logger.info(f"검색된 문서 수: {len(retrieved_docs)}")

                    # 문서 포맷팅
                    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
                else:
                    logger.info(f"메타 질문 감지: {user_input} - 검색 스킵")

                # LLM 연결
                llm = RemoteRunnable(DEFAULT_LLM_URL)

                # 프롬프트 구성 (최근 4개 메시지 = 2턴만 사용)
                recent_messages = msgs.messages[-4:] if msgs.messages else []
                full_prompt = build_prompt(context, is_meta_question, recent_messages, user_input)
                logger.info(f"Full prompt length: {len(full_prompt)}")

                # 응답 생성 (스트리밍)
                with st.chat_message("ai"):
                    chat_container = st.empty()
                    chunks = []

                    # 스트리밍 응답 수집
                    for chunk in llm.stream(full_prompt):
                        chunk_text = extract_text_from_chunk(chunk)
                        if chunk_text:
                            chunks.append(chunk_text)
                            chat_container.markdown("".join(chunks))

                    # 응답 정리 및 저장
                    final_response = "".join(chunks)
                    logger.info(f"Final response length: {len(final_response)}")

                    # 프롬프트 패턴 제거
                    final_response = clean_response_text(final_response)

                    # 정리된 응답 표시 및 히스토리 저장
                    if final_response:
                        chat_container.markdown(final_response)
                        msgs.add_user_message(user_input)
                        msgs.add_ai_message(final_response)
                    else:
                        logger.error("No response generated from LLM")
                        chat_container.error("LLM에서 응답을 생성하지 못했습니다.")

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
