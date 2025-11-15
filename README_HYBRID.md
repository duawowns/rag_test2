# Future Systems 회사소개 챗봇 - RAG + PEFT 하이브리드

RAG (Retrieval-Augmented Generation) + PEFT (Parameter-Efficient Fine-Tuning)를 결합한 하이브리드 회사소개 챗봇입니다.

## 주요 특징

- **RAG 시스템**: PDF 회사 소개 자료 + CSV 직원 데이터 200명
- **PEFT 파인튜닝**: LLM LoRA + 임베딩 모델 파인튜닝
- **하이브리드 검색**: 문서 검색 + 구조화된 데이터 검색

## 데이터셋

### 1. PDF 회사 소개 자료
- `futuresystems_company_brochure.pdf`: 회사 소개, 비전, 사업 분야

### 2. CSV 직원 데이터 (200명)
- `company_data.csv`: 이름, 직급, 부서, 전화번호, 이메일, 입사일, 담당업무

**특수 쿼리 예시:**
- "대표이사의 전화번호를 알려줘" → "010-777-7777"
- "염재준" 또는 "염재준 전화번호" → "010-3839-3418"

## 파일 구조

```
.
├── company_data.csv                      # 직원 데이터 200명
├── futuresystems_company_brochure.pdf    # 회사 소개 PDF
├── prepare_training_data_hybrid.py       # PDF + CSV 학습 데이터 생성
├── train_llm_lora.py                     # LLM LoRA 파인튜닝
├── train_embedding.py                    # 임베딩 모델 파인튜닝
├── company_intro_chatbot_hybrid.py       # 하이브리드 챗봇 (최종)
├── company_intro_chatbot_peft.py         # PEFT 전용 챗봇
├── company_intro_chatbot.py              # 기본 챗봇 (RemoteRunnable)
└── requirements.txt                       # 필요 패키지
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### Quick Start (하이브리드 챗봇)

```bash
streamlit run company_intro_chatbot_hybrid.py
```

사이드바에서:
1. "회사 직원 데이터 자동 로드" 체크 (기본 체크됨)
2. "문서 처리하기" 클릭
3. 질문 입력:
   - "대표이사의 전화번호를 알려줘"
   - "염재준"
   - "김철수는 무슨 일을 하나요?"

### 전체 파인튜닝 프로세스

#### 1단계: 하이브리드 학습 데이터 생성

```bash
python prepare_training_data_hybrid.py
```

**출력:**
- `training_data_hybrid.json`: PDF + CSV 통합 학습 데이터
- PDF 샘플: 회사 소개 내용
- CSV 샘플: 직원 정보별 7가지 질문 패턴

#### 2단계: 임베딩 모델 파인튜닝

```bash
python train_embedding.py
```

**출력:** `./embedding_finetuned_model/`

#### 3단계: LLM LoRA 파인튜닝

```bash
python train_llm_lora.py
```

**주의:** GPU 필요 (최소 16GB VRAM)

**출력:** `./llm_lora_model/`

#### 4단계: 챗봇 실행

```bash
streamlit run company_intro_chatbot_hybrid.py
```

## 학습 데이터 구조

### CSV 데이터에서 생성되는 질문 패턴 (직원 1명당 7가지)

1. 전화번호 질문: "{이름}의 전화번호를 알려주세요."
2. 간단 전화번호: "{이름} 전화번호"
3. 이메일 질문: "{이름}의 이메일 주소를 알려주세요."
4. 부서 질문: "{이름}은 어느 부서인가요?"
5. 직급 질문: "{이름}의 직급은 무엇인가요?"
6. 업무 질문: "{이름}은 무슨 일을 하나요?"
7. 종합 정보: "{이름}에 대해 알려주세요."

### 특수 쿼리 (대표이사)

- "대표이사의 전화번호를 알려주세요."
- "대표이사 전화번호"
- "CEO 전화번호는?"

**모두 동일 답변:** "010-777-7777"

## 하이브리드 챗봇 특징

### RAG 검색
- **PDF 검색**: 회사 비전, 사업 분야, 기술력 등
- **CSV 검색**: 직원 이름, 전화번호, 이메일, 부서 등
- **MMR 알고리즘**: 다양성 있는 검색 결과

### PEFT 파인튜닝 (선택)
- 사이드바에서 "PEFT 파인튜닝 모델 사용" 체크
- 로컬 LoRA 모델 사용 (느리지만 도메인 특화)
- 미체크 시 RemoteRunnable 사용 (빠름)

## 예시 쿼리

### 직원 정보 쿼리
```
Q: 대표이사의 전화번호를 알려줘
A: 010-777-7777

Q: 염재준
A: 010-3839-3418

Q: 김철수에 대해 알려주세요
A: 김철수님은 경영진 대표이사으로, 전사 경영 총괄을(를) 담당하고 있습니다.
   연락처는 010-777-7777, 이메일은 ceo@future.co.kr입니다.

Q: 개발부에는 누가 있나요?
A: 개발부에는 이민수 이사, 김태희 사원, 이병헌 대리, 원빈 과장, 공유 부장,
   하정우 차장, 송강호 이사, 최민식 상무 등이 있습니다.
```

### 회사 정보 쿼리
```
Q: Future Systems는 어떤 회사인가요?
A: [PDF에서 검색된 회사 소개 내용 반환]

Q: 회사의 주요 사업 분야는?
A: [PDF에서 검색된 사업 분야 내용 반환]
```

## 성능 최적화

### 검색 성능
- **search_type**: 'mmr' (Maximum Marginal Relevance)
- **k**: 5 (상위 5개 결과 반환)
- **fetch_k**: 10 (초기 10개 검색 후 MMR 적용)

### 모델 선택
- **빠른 응답**: RemoteRunnable (기본)
- **정확한 응답**: PEFT 로컬 모델 (GPU 필요)

## CSV 데이터 구조

```csv
이름,직급,부서,전화번호,이메일,입사일,담당업무
김철수,대표이사,경영진,010-777-7777,ceo@future.co.kr,2010-01-01,전사 경영 총괄
염재준,부사장,경영진,010-3839-3418,yjjoon@future.co.kr,2012-03-15,기술 총괄
...
```

**총 200명의 직원 데이터**

## 학습 데이터 통계

- **PDF 샘플**: ~50개 (회사 소개 청크)
- **CSV 샘플**: ~1,400개 (200명 × 7가지 질문)
- **총 학습 샘플**: ~1,450개

## 문제 해결

### CSV 로드 실패
```python
# company_data.csv 경로 확인
CSV_DATA_PATH = "company_data.csv"
```

### 검색 결과 부정확
- MMR 파라미터 조정:
  ```python
  search_kwargs={'k': 5, 'fetch_k': 10}
  ```

### PEFT 모델 메모리 부족
- RemoteRunnable 사용 (기본 옵션)
- 또는 배치 사이즈 줄이기

## 라이선스

MIT License
