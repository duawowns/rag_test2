import streamlit as st
import tiktoken
from loguru import logger
import torch

from langchain_core.messages import ChatMessage

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# PEFT 관련 import
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_community.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer

# 모델 경로 설정
LLM_LORA_PATH = "./llm_lora_model"
EMBEDDING_PATH = "./embedding_finetuned_model"
BASE_LLM_MODEL = "beomi/Llama-3-Open-Ko-8B"

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):

    doc_list = []

    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

class SentenceTransformerEmbeddings:
    """SentenceTransformer를 LangChain에서 사용하기 위한 래퍼"""
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

def get_vectorstore(text_chunks):
    # 파인튜닝된 임베딩 모델 사용
    embeddings = SentenceTransformerEmbeddings(EMBEDDING_PATH)
    vectordb = FAISS.from_documents(text_chunks, embeddings)

    return vectordb

@st.cache_resource
def load_peft_model():
    """파인튜닝된 LoRA 모델 로드 (캐싱)"""
    st.info("파인튜닝된 모델 로드 중...")

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(LLM_LORA_PATH)

    # 베이스 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_LLM_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # LoRA 어댑터 로드
    model = PeftModel.from_pretrained(base_model, LLM_LORA_PATH)

    # Pipeline 생성
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

    # LangChain LLM으로 래핑
    llm = HuggingFacePipeline(pipeline=pipe)

    st.success("모델 로드 완료!")
    return llm

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

def main():

    global retriever

    st.set_page_config(
    page_title="챗봇",
    page_icon="")

    st.title(":blue[챗봇]")
    st.caption("🤖 PEFT 파인튜닝 적용 버전")

    if "messages" not in st.session_state:
       st.session_state["messages"] = []

   #채팅 대화기록을 점검
    if "store" not in st.session_state:
       st.session_state["store"] =dict()


    def print_history():
        for msg in st.session_state.messages:
            st.chat_message(msg.role).write(msg.content)

    def add_history(role, content):
        st.session_state.messages.append(ChatMessage(role=role, content=content))

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    with st.sidebar:
        st.header("자료 업로드")
        uploaded_files =  st.file_uploader("PDF, DOCX, PPTX 파일을 업로드하세요",
                                          type=['pdf','docx','pptx'],
                                          accept_multiple_files=True)
        process = st.button("문서 처리하기")

    if process:

        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)
        retriever = vectorstore.as_retriever(search_type = 'mmr', vervose = True)
        st.session_state['retriever'] =retriever

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                        "content": "안녕하세요! Future Systems 회사소개 챗봇입니다. 회사에 대해 궁금하신 점을 물어보세요!"}]

    def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
        return "\n\n".join(doc.page_content for doc in docs)

    RAG_PROMPT_TEMPLATE = """당신은 Future Systems 회사 소개 전문 AI 어시스턴트입니다.
                             검색된 문서 내용을 바탕으로 회사에 대한 질문에 친절하고 정확하게 답변해주세요.
                             답변은 구체적이고 상세하게 작성하되, 문서에 없는 내용은 추측하지 말고 모른다고 답변하세요.

                             Question: {question}
                             Context: {context}
                             Answer:"""

    print_history()

    if user_input := st.chat_input("궁금한 점을 물어보세요..."):
        #사용자가 입력한 내용
        add_history("user", user_input)
        st.chat_message("user").write(f"{user_input}")
        with st.chat_message("assistant"):

            # 파인튜닝된 로컬 LLM 사용
            llm = load_peft_model()
            chat_container = st.empty()

            if  st.session_state.processComplete==True:
                prompt1 = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

                retriever = st.session_state['retriever']
               # 체인을 생성합니다.
                rag_chain = (
                   {
                       "context": retriever | format_docs,
                       "question": RunnablePassthrough(),
                   }
                   | prompt1
                   | llm
                )

                # 스트리밍 대신 일반 호출
                answer = rag_chain.invoke(user_input)
                response_text = extract_text_from_chunk(answer)
                chat_container.markdown(response_text)
                add_history("ai", response_text)

            else:
                prompt2 = ChatPromptTemplate.from_template(
                    "다음 질문에 대해 간단히 답변해주세요. 만약 회사 관련 질문이라면, 먼저 회사 소개 자료를 업로드하도록 안내해주세요:\n{input}"
                )

                # 체인을 생성합니다.
                chain = prompt2 | llm

                answer = chain.invoke(user_input)
                response_text = extract_text_from_chunk(answer)
                chat_container.markdown(response_text)
                add_history("ai", response_text)

if __name__ == '__main__':
    main()
