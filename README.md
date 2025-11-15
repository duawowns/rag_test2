# Future Systems 회사소개 챗봇

RAG 기반 회사소개 챗봇 - 직원 정보 + 회사 소개 자료

## 빠른 시작

### 원클릭 실행

```bash
./run_chatbot.sh
```

또는

```bash
streamlit run chatbot.py
```

## 기능

- ✅ 직원 정보 자동 검색 (CSV)
- ✅ 회사 소개 자료 검색 (PDF)
- ✅ 자동 데이터 로드
- ✅ RAG 기반 정확한 답변

## 예시 질문

```
Q: 대표이사의 전화번호를 알려줘
A: 010-777-7777

Q: 염재준
A: 010-3839-3418

Q: 정원규는 무슨 일을 하나요?
A: 정원규님은 퓨쳐시스템 대표이사으로, 전사 경영 총괄을(를) 담당하고 있습니다.
```

## 데이터

### 직원 정보 (company_data.csv) - 2명
- 정원규 (대표이사) - 010-777-7777
- 염재준 (사원, VPN팀) - 010-3839-3418

### 회사 소개 (company_info_data.csv) - 200개 항목
- 회사 기본정보 10개
- VPN 제품 30개
- 보안 솔루션 30개
- 클라우드 서비스 20개
- 주요 고객사 30개
- 기술 인증 20개
- 프로젝트 실적 30개
- 기술 스택 20개
- 수상 실적 10개

### 회사 소개 PDF (futuresystems_company_brochure.pdf)
- 회사 비전, 사업 분야, 기술력 등

## ngrok URL 변경

사이드바의 "LLM 서버 URL"에서 새 URL을 입력하세요.

## 파일 구조

```
.
├── chatbot.py                  # 메인 챗봇 (간소화 버전)
├── run_chatbot.sh              # 실행 스크립트
├── company_data.csv            # 직원 데이터
├── futuresystems_company_brochure.pdf  # 회사 소개
└── requirements.txt            # 필요 패키지
```

## 문제 해결

### LLM 서버 연결 실패
1. ngrok이 실행 중인지 확인
2. 사이드바에서 URL 확인 및 업데이트

### 데이터 로드 실패
- CSV, PDF 파일이 같은 폴더에 있는지 확인
