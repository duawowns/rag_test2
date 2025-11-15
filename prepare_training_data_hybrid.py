"""
PDF + CSV 데이터에서 학습 데이터를 추출하여 PEFT 학습용 데이터셋 생성
"""
import json
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def create_training_data_from_pdf(pdf_path):
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

    return training_data

def create_training_data_from_csv(csv_path):
    """CSV에서 학습 데이터 생성 (직원 정보)"""

    # CSV 로드
    df = pd.read_csv(csv_path)

    training_data = []

    # 각 직원에 대해 Q&A 생성
    for idx, row in df.iterrows():
        name = row['이름']
        position = row['직급']
        department = row['부서']
        phone = row['전화번호']
        email = row['이메일']
        join_date = row['입사일']
        role = row['담당업무']

        # 1. 전화번호 질문
        training_data.append({
            "instruction": f"{name}의 전화번호를 알려주세요.",
            "input": "",
            "output": phone
        })

        training_data.append({
            "instruction": f"{name} 전화번호",
            "input": "",
            "output": phone
        })

        # 2. 직급별 특수 질문 (대표이사)
        if "대표" in position or "CEO" in position:
            training_data.append({
                "instruction": "대표이사의 전화번호를 알려주세요.",
                "input": "",
                "output": phone
            })
            training_data.append({
                "instruction": "대표이사 전화번호",
                "input": "",
                "output": phone
            })
            training_data.append({
                "instruction": "CEO 전화번호는?",
                "input": "",
                "output": phone
            })

        # 3. 이메일 질문
        training_data.append({
            "instruction": f"{name}의 이메일 주소를 알려주세요.",
            "input": "",
            "output": email
        })

        # 4. 부서 질문
        training_data.append({
            "instruction": f"{name}은 어느 부서인가요?",
            "input": "",
            "output": f"{name}님은 {department} 소속입니다."
        })

        # 5. 직급 질문
        training_data.append({
            "instruction": f"{name}의 직급은 무엇인가요?",
            "input": "",
            "output": f"{name}님의 직급은 {position}입니다."
        })

        # 6. 담당 업무 질문
        training_data.append({
            "instruction": f"{name}은 무슨 일을 하나요?",
            "input": "",
            "output": f"{name}님은 {role}을(를) 담당하고 있습니다."
        })

        # 7. 종합 정보 질문
        training_data.append({
            "instruction": f"{name}에 대해 알려주세요.",
            "input": "",
            "output": f"{name}님은 {department} {position}으로, {role}을(를) 담당하고 있습니다. 연락처는 {phone}, 이메일은 {email}입니다."
        })

    return training_data

def main():
    pdf_path = "futuresystems_company_brochure.pdf"
    csv_path = "company_data.csv"
    output_path = "training_data_hybrid.json"

    print("=" * 50)
    print("하이브리드 학습 데이터 생성")
    print("=" * 50)

    # PDF 데이터
    print("\n1. PDF 데이터 처리 중...")
    pdf_data = create_training_data_from_pdf(pdf_path)
    print(f"   PDF 샘플: {len(pdf_data)}개")

    # CSV 데이터
    print("\n2. CSV 데이터 처리 중...")
    csv_data = create_training_data_from_csv(csv_path)
    print(f"   CSV 샘플: {len(csv_data)}개")

    # 통합
    all_data = pdf_data + csv_data

    # JSON으로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\n총 학습 데이터: {len(all_data)}개 샘플")
    print(f"저장 경로: {output_path}")
    print("=" * 50)

    # 샘플 출력
    print("\n=== PDF 샘플 ===")
    print(json.dumps(pdf_data[0], ensure_ascii=False, indent=2))

    print("\n=== CSV 샘플 (대표이사) ===")
    for item in csv_data:
        if "대표이사" in item["instruction"]:
            print(json.dumps(item, ensure_ascii=False, indent=2))
            break

if __name__ == "__main__":
    main()
