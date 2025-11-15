#!/bin/bash

echo "========================================="
echo "Future Systems 챗봇 시작"
echo "========================================="
echo ""

# 필요한 파일 확인
if [ ! -f "company_data.csv" ]; then
    echo "❌ company_data.csv 파일이 없습니다."
    exit 1
fi

if [ ! -f "futuresystems_company_brochure.pdf" ]; then
    echo "❌ futuresystems_company_brochure.pdf 파일이 없습니다."
    exit 1
fi

echo "✅ 데이터 파일 확인 완료"
echo ""

# Streamlit 실행
echo "🚀 챗봇 실행 중..."
echo ""
echo "브라우저가 자동으로 열립니다."
echo "종료하려면 Ctrl+C를 누르세요."
echo ""

streamlit run chatbot.py
