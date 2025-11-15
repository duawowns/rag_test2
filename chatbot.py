"""
완벽한 RAG + PEFT + History 하이브리드 챗봇
- RAG: FAISS 벡터 검색 (회사 데이터)
- PEFT: 외부 Llama3 LLM 서버 (ngrok)
- History: LangChain StreamlitChatMessageHistory (자동)
"""

import streamlit as st
import tiktoken
from loguru import logger
import pandas as pd
import re
from typing import List, Tuple, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langserve import RemoteRunnable

# ==================== 설정 ====================
CSV_EMPLOYEE_PATH = "company_data.csv"
CSV_COMPANY_INFO_PATH = "company_info_data.csv"
PDF_DATA_PATH = "futuresystems_company_brochure.pdf"
DEFAULT_LLM_URL = "https://dioramic-corrin-undetractively.ngrok-free.dev/llm/"

# ==================== RAG System ====================
class RAGSystem:
    """RAG 시스템: 데이터 로드 및 검색"""

    @staticmethod
    def tiktoken_len(text: str) -> int:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(tokenizer.encode(text))

    @staticmethod
    def load_employee_csv(csv_path: str) -> List[Document]:
        """직원 CSV → Documents"""
        df = pd.read_csv(csv_path)
        documents = []

        for _, row in df.iterrows():
            content = f"""이름: {row['이름']}
직급: {row['직급']}
부서: {row['부서']}
전화번호: {row['전화번호']}
이메일: {row['이메일']}
입사일: {row['입사일']}
담당업무: {row['담당업무']}"""

            doc = Document(
                page_content=content,
                metadata={"source": "employee", "name": row['이름']}
            )
            documents.append(doc)

        return documents

    @staticmethod
    def load_company_info_csv(csv_path: str) -> List[Document]:
        """회사 정보 CSV → Documents"""
        df = pd.read_csv(csv_path)
        documents = []

        for _, row in df.iterrows():
            content = f"[{row['카테고리']}] {row['항목']}\n{row['내용']}"
            doc = Document(
                page_content=content,
                metadata={"source": "company_info", "category": row['카테고리']}
            )
            documents.append(doc)

        return documents

    @staticmethod
    def load_pdf(pdf_path: str) -> List[Document]:
        """PDF → Chunks"""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=100,
            length_function=RAGSystem.tiktoken_len
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    @staticmethod
    @st.cache_resource
    def initialize():
        """RAG 시스템 초기화 (캐싱)"""
        all_docs = []

        # 데이터 로드
        try:
            all_docs.extend(RAGSystem.load_employee_csv(CSV_EMPLOYEE_PATH))
            logger.info(f"직원 데이터 로드 완료")
        except Exception as e:
            logger.error(f"직원 CSV 로드 실패: {e}")

        try:
            all_docs.extend(RAGSystem.load_company_info_csv(CSV_COMPANY_INFO_PATH))
            logger.info(f"회사 정보 데이터 로드 완료")
        except Exception as e:
            logger.error(f"회사 정보 CSV 로드 실패: {e}")

        try:
            all_docs.extend(RAGSystem.load_pdf(PDF_DATA_PATH))
            logger.info(f"PDF 데이터 로드 완료")
        except Exception as e:
            logger.error(f"PDF 로드 실패: {e}")

        if not all_docs:
            logger.error("로드된 문서가 없습니다")
            return None

        # 벡터스토어 생성
        embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = FAISS.from_documents(all_docs, embeddings)
        retriever = vectorstore.as_retriever(
            search_type='mmr',
            search_kwargs={'k': 10, 'fetch_k': 20}
        )

        logger.info(f"RAG 시스템 초기화 완료: {len(all_docs)} documents")
        return retriever


# ==================== Query Processor ====================
class QueryProcessor:
    """질문 처리: 분류, 재작성, 엔티티 추출"""

    @staticmethod
    def extract_entity(messages) -> Optional[str]:
        """최근 엔티티 추출 (사람 이름, 회사명)"""
        recent = messages[-4:] if len(messages) >= 4 else messages
        entities = []

        for msg in reversed(recent):
            if msg.type == "human":
                # 한글 이름 패턴
                names = re.findall(r'[가-힣]{2,4}(?=의|이|은|는|를|을|에게|한테)', msg.content)
                entities.extend(names)

                # 회사명
                if "퓨쳐시스템" in msg.content or "Future Systems" in msg.content:
                    entities.append("퓨쳐시스템")

        return entities[0] if entities else None

    @staticmethod
    def rewrite_query(user_input: str, messages) -> str:
        """Query Rewriting: "전화번호는?" → "염재준의 전화번호는?"""
        entity = QueryProcessor.extract_entity(messages)
        if not entity:
            return user_input

        # 짧은 속성 질문 감지
        patterns = ["전화번호", "이메일", "직급", "부서", "담당업무", "주소", "연락처"]
        for pattern in patterns:
            if pattern in user_input and len(user_input) <= 15:
                return f"{entity}의 {user_input}"

        return user_input

    @staticmethod
    def classify_question(user_input: str, has_history: bool) -> Tuple[str, bool]:
        """
        질문 분류
        Returns: (question_type, is_followup)
            - question_type: "meta" | "normal"
            - is_followup: True/False
        """
        # 메타 질문 키워드
        meta_keywords = ["이전", "그", "그사람", "방금", "아까", "위", "앞서"]
        lang_keywords = ["영어", "일본어", "중국어", "한국어", "번역", "translate", "요약", "summarize", "변경"]

        has_meta = any(k in user_input for k in meta_keywords)
        has_lang = any(k in user_input for k in lang_keywords)

        if has_meta or (has_lang and has_history):
            return "meta", False

        # Follow-up 질문 (짧고 히스토리 있음)
        is_followup = len(user_input.split()) <= 3 and has_history and not has_lang

        return "normal", is_followup


# ==================== Prompt Builder ====================
class PromptBuilder:
    """프롬프트 생성 (Llama3 최적화)"""

    @staticmethod
    def build_meta_prompt(user_input: str, history_messages) -> str:
        """메타 질문 프롬프트 (번역, 요약 등)"""
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

Conversation History:
"""

        # 히스토리 추가
        for msg in history_messages:
            role = "User" if msg.type == "human" else "Assistant"
            prompt += f"{role}: {msg.content}\n"

        prompt += f"\nUser Request: {user_input}\n\nResponse:"
        return prompt

    @staticmethod
    def build_normal_prompt(user_input: str, context: str, history_messages) -> str:
        """일반 질문 프롬프트"""
        prompt = """당신은 Future Systems 회사 소개 전문 AI 어시스턴트입니다.

검색된 회사 정보가 질문과 관련이 있다면 그 정보를 우선적으로 사용하여 정확하게 답변하세요.
검색된 정보가 질문과 관련이 없거나 충분하지 않다면 일반 지식을 사용하여 친절하게 답변하세요.
특히 회사 직원의 전화번호, 이메일, 주소 등은 검색된 정보를 정확히 제공하세요.

"""

        # RAG Context
        if context:
            prompt += f"검색된 회사 정보:\n{context}\n---\n\n"

        # 최근 히스토리 (최근 2턴 = 4개 메시지)
        if history_messages:
            recent = history_messages[-4:]
            prompt += "최근 대화:\n"
            for msg in recent:
                role = "사용자" if msg.type == "human" else "AI"
                prompt += f"{role}: {msg.content}\n"
            prompt += "\n"

        # 현재 질문
        prompt += f"현재 질문: {user_input}\n\n답변:"
        return prompt


# ==================== Response Handler ====================
class ResponseHandler:
    """응답 처리: 스트리밍, 정리, 저장"""

    @staticmethod
    def extract_text_from_chunk(chunk) -> str:
        """Chunk에서 텍스트 추출"""
        if isinstance(chunk, str):
            return chunk
        elif isinstance(chunk, dict):
            for key in ['content', 'text', 'output', 'answer', 'response']:
                if key in chunk and chunk[key]:
                    return str(chunk[key])
        elif hasattr(chunk, 'content'):
            content = chunk.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list) and content:
                return str(content[0])
            return str(content) if content else ''
        elif hasattr(chunk, 'text'):
            return str(chunk.text) if chunk.text else ''
        else:
            try:
                return str(chunk)
            except:
                return ''
        return ''

    @staticmethod
    def clean_response(response: str) -> str:
        """응답 정리: 프롬프트 패턴 제거"""
        response = response.strip()

        # 패턴 제거
        patterns = [
            (r'^[QA]:\s*', ''),
            (r'\n[QA]:\s*', '\n'),
            (r'^(사용자|어시스턴트|AI|User|Assistant):\s*', ''),
            (r'\n(사용자|어시스턴트|AI|User|Assistant):\s*', '\n'),
            (r'^답변\s*[:：]\s*', ''),
            (r'현재.*?질문\s*[:：].*?\n', ''),
            (r'Response\s*:\s*', ''),
        ]

        for pattern, replacement in patterns:
            response = re.sub(pattern, replacement, response, flags=re.MULTILINE)

        # 중복 공백 정리
        response = re.sub(r'\n\s*\n', '\n\n', response)
        response = re.sub(r'\n{3,}', '\n\n', response)

        return response.strip()


# ==================== Main Application ====================
def main():
    st.set_page_config(page_title="챗봇", page_icon="")
    st.title(":blue[챗봇]")

    # 1. 초기화
    msgs = StreamlitChatMessageHistory(key="chat_messages")
    retriever = RAGSystem.initialize()

    if not retriever:
        st.error("RAG 시스템 초기화 실패")
        return

    # 2. 대화 기록 표시
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # 3. 사용자 입력 처리
    if user_input := st.chat_input("질문을 입력하세요"):
        st.chat_message("human").write(user_input)

        try:
            # 3-1. 질문 분류
            question_type, is_followup = QueryProcessor.classify_question(
                user_input, bool(msgs.messages)
            )

            # 3-2. 검색 (일반 질문만)
            context = ""
            if question_type == "normal":
                # Query Rewriting
                search_query = user_input
                if is_followup:
                    rewritten = QueryProcessor.rewrite_query(user_input, msgs.messages)
                    if rewritten != user_input:
                        search_query = rewritten
                        logger.info(f"Query Rewriting: '{user_input}' → '{search_query}'")

                # 검색 실행
                docs = retriever.invoke(search_query)
                context = "\n\n".join([doc.page_content for doc in docs])
                logger.info(f"검색 완료: {len(docs)} documents")
            else:
                logger.info(f"메타 질문 감지 - 검색 스킵")

            # 3-3. 프롬프트 생성
            if question_type == "meta":
                prompt = PromptBuilder.build_meta_prompt(user_input, msgs.messages)
            else:
                prompt = PromptBuilder.build_normal_prompt(user_input, context, msgs.messages)

            logger.info(f"프롬프트 길이: {len(prompt)}")

            # 3-4. LLM 호출 (스트리밍)
            llm = RemoteRunnable(DEFAULT_LLM_URL)

            with st.chat_message("ai"):
                chat_container = st.empty()
                chunks = []

                for chunk in llm.stream(prompt):
                    chunk_text = ResponseHandler.extract_text_from_chunk(chunk)
                    if chunk_text:
                        chunks.append(chunk_text)
                        chat_container.markdown("".join(chunks))

                # 3-5. 응답 정리 및 저장
                final_response = "".join(chunks)
                final_response = ResponseHandler.clean_response(final_response)

                if final_response:
                    chat_container.markdown(final_response)
                    msgs.add_user_message(user_input)
                    msgs.add_ai_message(final_response)
                    logger.info("응답 저장 완료")
                else:
                    chat_container.error("LLM에서 응답을 생성하지 못했습니다.")

        except Exception as e:
            logger.error(f"오류 발생: {e}")
            st.chat_message("ai").error(f"오류가 발생했습니다: {str(e)}")


if __name__ == '__main__':
    main()
