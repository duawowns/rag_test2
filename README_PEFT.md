# Future Systems 회사소개 챗봇 - PEFT 버전

PEFT (Parameter-Efficient Fine-Tuning)를 적용한 회사소개 챗봇입니다.

## 주요 특징

- **LLM LoRA 파인튜닝**: Llama3 한국어 모델을 회사 데이터로 파인튜닝
- **임베딩 모델 파인튜닝**: 회사 도메인 특화 임베딩 모델
- **RAG 시스템**: 파인튜닝된 모델 + 문서 검색

## 파일 구조

```
.
├── company_intro_chatbot.py          # 기본 챗봇 (RemoteRunnable 사용)
├── company_intro_chatbot_peft.py     # PEFT 적용 챗봇
├── prepare_training_data.py          # 학습 데이터 생성 스크립트
├── train_llm_lora.py                 # LLM LoRA 파인튜닝 스크립트
├── train_embedding.py                # 임베딩 모델 파인튜닝 스크립트
├── futuresystems_company_brochure.pdf # 회사 소개 자료
└── requirements.txt                   # 필요 패키지
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 1단계: 학습 데이터 생성

```bash
python prepare_training_data.py
```

출력: `training_data.json`

### 2단계: 임베딩 모델 파인튜닝

```bash
python train_embedding.py
```

출력: `./embedding_finetuned_model/`

### 3단계: LLM LoRA 파인튜닝

```bash
python train_llm_lora.py
```

출력: `./llm_lora_model/`

**주의**:
- LLM 파인튜닝은 GPU가 필요합니다 (최소 16GB VRAM 권장)
- CPU로도 가능하지만 매우 느립니다

### 4단계: 챗봇 실행

**PEFT 버전 (파인튜닝 모델 사용)**
```bash
streamlit run company_intro_chatbot_peft.py
```

**기본 버전 (RemoteRunnable 사용)**
```bash
streamlit run company_intro_chatbot.py
```

## 모델 정보

### LLM
- **베이스 모델**: beomi/Llama-3-Open-Ko-8B
- **파인튜닝 방법**: LoRA (Low-Rank Adaptation)
- **LoRA 설정**:
  - rank: 8
  - alpha: 32
  - target_modules: q_proj, v_proj

### 임베딩 모델
- **베이스 모델**: jhgan/ko-sroberta-multitask
- **파인튜닝 방법**: Cosine Similarity Loss
- **학습 데이터**: PDF 청크 기반 query-passage pairs

## 학습 설정

### LLM LoRA
- Epochs: 3
- Batch Size: 2
- Gradient Accumulation: 4
- Learning Rate: 2e-4
- FP16: True

### 임베딩 모델
- Epochs: 3
- Batch Size: 16
- Warmup Steps: 100

## 성능 비교

| 항목 | 기본 버전 | PEFT 버전 |
|------|----------|----------|
| LLM | 원격 서버 | 로컬 파인튜닝 모델 |
| 임베딩 | 사전학습 모델 | 도메인 특화 모델 |
| 응답 품질 | 일반적 | 회사 특화 |
| 검색 정확도 | 보통 | 향상됨 |
| 추론 속도 | 빠름 (서버) | 느림 (로컬) |

## 문제 해결

### GPU 메모리 부족
```python
# train_llm_lora.py에서 배치 사이즈 줄이기
per_device_train_batch_size=1
```

### CPU 사용
```python
# device_map을 "cpu"로 변경
device_map="cpu"
```

## 추가 개선 사항

1. **더 많은 학습 데이터**: 회사 FAQ, 블로그, 보도자료 등
2. **QLoRA**: 4bit 양자화로 메모리 절약
3. **하이퍼파라미터 튜닝**: LoRA rank, learning rate 최적화
4. **평가 지표**: BLEU, ROUGE 등으로 성능 측정

## 라이선스

MIT License
