"""
PDF에서 학습 데이터를 추출하여 PEFT 학습용 데이터셋 생성
"""
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def create_training_data(pdf_path, output_path):
    """PDF에서 학습 데이터 생성"""

    # PDF 로드
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 텍스트 청크로 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(documents)

    # 학습 데이터 생성
    training_data = []

    # 회사 정보 관련 질문 템플릿
    question_templates = [
        "Future Systems에 대해 알려주세요.",
        "회사 소개를 해주세요.",
        "회사의 주요 사업은 무엇인가요?",
        "회사의 비전은 무엇인가요?",
        "회사의 서비스에 대해 설명해주세요.",
        "회사의 기술력은 어떤가요?",
        "회사의 강점은 무엇인가요?",
    ]

    # 각 청크를 기반으로 instruction-output 생성
    for i, chunk in enumerate(chunks):
        content = chunk.page_content.strip()

        if len(content) < 50:  # 너무 짧은 청크는 스킵
            continue

        # 여러 질문 템플릿 사용 (데이터 다양성)
        template_idx = i % len(question_templates)
        question = question_templates[template_idx]

        # Instruction 형식으로 데이터 생성
        instruction = f"""당신은 Future Systems 회사 소개 전문 AI 어시스턴트입니다.
검색된 문서 내용을 바탕으로 회사에 대한 질문에 친절하고 정확하게 답변해주세요.

질문: {question}
문서 내용: {content}"""

        output = content

        training_data.append({
            "instruction": instruction,
            "input": "",
            "output": output
        })

    # JSON으로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    print(f"학습 데이터 생성 완료: {len(training_data)}개의 샘플")
    print(f"저장 경로: {output_path}")

    return training_data

if __name__ == "__main__":
    pdf_path = "futuresystems_company_brochure.pdf"
    output_path = "training_data.json"

    data = create_training_data(pdf_path, output_path)

    # 샘플 출력
    print("\n=== 샘플 데이터 ===")
    print(json.dumps(data[0], ensure_ascii=False, indent=2))
