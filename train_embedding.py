"""
임베딩 모델 파인튜닝 스크립트
"""
import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

# 설정
BASE_MODEL = "jhgan/ko-sroberta-multitask"
OUTPUT_DIR = "./embedding_finetuned_model"
PDF_PATH = "futuresystems_company_brochure.pdf"
EPOCHS = 3
BATCH_SIZE = 16

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def prepare_training_pairs(pdf_path):
    """PDF에서 학습용 sentence pair 생성"""

    print(f"PDF 로드: {pdf_path}")
    # PDF 로드
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 텍스트 청크로 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(documents)

    # 질문 템플릿
    query_templates = [
        "Future Systems 회사 정보",
        "회사 소개",
        "사업 분야",
        "회사 서비스",
        "기술력",
        "회사 비전",
        "회사 강점",
        "제공 서비스",
        "솔루션",
        "고객사"
    ]

    # InputExample 생성 (query-passage pair)
    train_examples = []

    for i, chunk in enumerate(chunks):
        content = chunk.page_content.strip()

        if len(content) < 50:  # 너무 짧은 청크는 스킵
            continue

        # 각 청크에 대해 여러 쿼리 매칭
        for template in query_templates:
            # positive pair: 질문과 관련 문서
            example = InputExample(
                texts=[template, content],
                label=1.0  # similarity score
            )
            train_examples.append(example)

    print(f"학습 페어 생성 완료: {len(train_examples)} 쌍")
    return train_examples

def train_embedding_model():
    """임베딩 모델 파인튜닝"""

    print("=" * 50)
    print("임베딩 모델 파인튜닝 시작")
    print("=" * 50)

    # 1. 베이스 모델 로드
    print(f"\n1. 베이스 모델 로드: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL)

    # 2. 학습 데이터 준비
    print("\n2. 학습 데이터 준비")
    train_examples = prepare_training_pairs(PDF_PATH)

    # 3. DataLoader 생성
    print("\n3. DataLoader 생성")
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    # 4. Loss 함수 설정 (CosineSimilarityLoss)
    print("\n4. Loss 함수 설정")
    train_loss = losses.CosineSimilarityLoss(model)

    # 5. 학습 시작
    print("\n5. 학습 시작")
    print("=" * 50)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=100,
        output_path=OUTPUT_DIR,
        show_progress_bar=True
    )

    # 6. 모델 저장
    print("\n6. 모델 저장")
    model.save(OUTPUT_DIR)

    print("\n" + "=" * 50)
    print(f"학습 완료! 모델 저장 위치: {OUTPUT_DIR}")
    print("=" * 50)

    # 7. 테스트
    print("\n7. 임베딩 테스트")
    test_query = "Future Systems 회사 소개"
    embedding = model.encode(test_query)
    print(f"   쿼리: {test_query}")
    print(f"   임베딩 차원: {len(embedding)}")
    print(f"   임베딩 샘플: {embedding[:5]}")

if __name__ == "__main__":
    train_embedding_model()
