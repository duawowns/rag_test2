"""
퓨쳐시스템 회사 소개 데이터 200개 항목 생성
"""
import csv

def generate_company_data():
    """회사 소개 관련 200개 데이터 항목 생성"""
    data = []

    # 1. 회사 기본 정보 (10개)
    basic_info = [
        {"카테고리": "회사정보", "항목": "회사명", "내용": "퓨쳐시스템"},
        {"카테고리": "회사정보", "항목": "설립연도", "내용": "2010년"},
        {"카테고리": "회사정보", "항목": "대표이사", "내용": "정원규"},
        {"카테고리": "회사정보", "항목": "본사위치", "내용": "서울특별시 강남구 테헤란로"},
        {"카테고리": "회사정보", "항목": "직원수", "내용": "200명"},
        {"카테고리": "회사정보", "항목": "자본금", "내용": "100억원"},
        {"카테고리": "회사정보", "항목": "주요사업", "내용": "VPN 솔루션, 보안 서비스, 클라우드 인프라"},
        {"카테고리": "회사정보", "항목": "비전", "내용": "안전한 네트워크 세상 구현"},
        {"카테고리": "회사정보", "항목": "미션", "내용": "혁신적인 보안 기술로 고객 가치 창출"},
        {"카테고리": "회사정보", "항목": "핵심가치", "내용": "혁신, 신뢰, 고객만족"},
    ]
    data.extend(basic_info)

    # 2. VPN 제품/서비스 (30개)
    vpn_products = []
    for i in range(1, 31):
        vpn_products.append({
            "카테고리": "VPN제품",
            "항목": f"VPN 솔루션 {i}",
            "내용": f"기업용 VPN 솔루션 - SSL VPN, IPSec VPN 지원, 최대 {i*100}명 동시 접속, AES-256 암호화"
        })
    data.extend(vpn_products)

    # 3. 보안 솔루션 (30개)
    security_solutions = []
    security_types = ["방화벽", "침입탐지", "DDoS 방어", "웹방화벽", "DB보안", "엔드포인트보안"]
    for i in range(1, 31):
        sec_type = security_types[i % len(security_types)]
        security_solutions.append({
            "카테고리": "보안솔루션",
            "항목": f"{sec_type} 솔루션 {i}",
            "내용": f"{sec_type} 전문 솔루션 - 실시간 모니터링, 자동 차단, 위협 분석 기능"
        })
    data.extend(security_solutions)

    # 4. 클라우드 서비스 (20개)
    cloud_services = []
    for i in range(1, 21):
        cloud_services.append({
            "카테고리": "클라우드",
            "항목": f"클라우드 서비스 {i}",
            "내용": f"프라이빗/퍼블릭/하이브리드 클라우드 지원, 자동 스케일링, 백업/복구 기능"
        })
    data.extend(cloud_services)

    # 5. 주요 고객사 (30개)
    customers = []
    industries = ["금융", "제조", "공공", "의료", "교육", "유통", "통신", "서비스", "IT", "건설"]
    for i in range(1, 31):
        industry = industries[i % len(industries)]
        customers.append({
            "카테고리": "고객사",
            "항목": f"{industry}업 고객사 {i}",
            "내용": f"{industry}업 대기업 VPN 및 보안 솔루션 구축 - {2020+i%5}년 계약"
        })
    data.extend(customers)

    # 6. 기술 인증 (20개)
    certifications = [
        {"카테고리": "인증", "항목": "ISO 27001", "내용": "정보보안 관리체계 국제 인증"},
        {"카테고리": "인증", "항목": "ISO 9001", "내용": "품질경영시스템 인증"},
        {"카테고리": "인증", "항목": "GS 인증", "내용": "우수 소프트웨어 인증"},
        {"카테고리": "인증", "항목": "CC 인증", "내용": "국제 공통평가기준 인증"},
        {"카테고리": "인증", "항목": "ISMS", "내용": "정보보호 관리체계 인증"},
        {"카테고리": "인증", "항목": "특허 1", "내용": "VPN 암호화 기술 특허"},
        {"카테고리": "인증", "항목": "특허 2", "내용": "네트워크 보안 특허"},
        {"카테고리": "인증", "항목": "특허 3", "내용": "클라우드 보안 특허"},
        {"카테고리": "인증", "항목": "특허 4", "내용": "AI 기반 위협 탐지 특허"},
        {"카테고리": "인증", "항목": "특허 5", "내용": "제로트러스트 보안 특허"},
        {"카테고리": "인증", "항목": "벤처기업 인증", "내용": "기술혁신형 벤처기업"},
        {"카테고리": "인증", "항목": "이노비즈 인증", "내용": "기술혁신 중소기업"},
        {"카테고리": "인증", "항목": "메인비즈 인증", "내용": "경영혁신 중소기업"},
        {"카테고리": "인증", "항목": "PCI-DSS", "내용": "결제카드 보안 표준"},
        {"카테고리": "인증", "항목": "GDPR 준수", "내용": "EU 개인정보보호 규정 준수"},
        {"카테고리": "인증", "항목": "SOC 2", "내용": "서비스 조직 통제 인증"},
        {"카테고리": "인증", "항목": "K-ISMS", "내용": "한국 정보보호 관리체계"},
        {"카테고리": "인증", "항목": "AWS 파트너", "내용": "AWS 공인 파트너"},
        {"카테고리": "인증", "항목": "Azure 파트너", "내용": "Microsoft Azure 파트너"},
        {"카테고리": "인증", "항목": "GCP 파트너", "내용": "Google Cloud 파트너"},
    ]
    data.extend(certifications)

    # 7. 프로젝트 실적 (30개)
    projects = []
    for i in range(1, 31):
        year = 2015 + (i % 10)
        projects.append({
            "카테고리": "프로젝트",
            "항목": f"{year}년 프로젝트 {i}",
            "내용": f"대기업 VPN 인프라 구축 프로젝트 - {i*10}억원 규모, {i*3}개월 소요"
        })
    data.extend(projects)

    # 8. 기술 스택 (20개)
    tech_stack = [
        {"카테고리": "기술", "항목": "VPN 프로토콜", "내용": "OpenVPN, IPSec, SSL/TLS, WireGuard"},
        {"카테고리": "기술", "항목": "암호화", "내용": "AES-256, RSA-4096, SHA-256"},
        {"카테고리": "기술", "항목": "인증", "내용": "LDAP, Active Directory, RADIUS, SAML 2.0"},
        {"카테고리": "기술", "항목": "방화벽", "내용": "Next-Gen Firewall, WAF, Network Firewall"},
        {"카테고리": "기술", "항목": "IDS/IPS", "내용": "Snort, Suricata 기반 침입탐지/차단"},
        {"카테고리": "기술", "항목": "SIEM", "내용": "실시간 보안 이벤트 모니터링"},
        {"카테고리": "기술", "항목": "DLP", "내용": "데이터 유출 방지 솔루션"},
        {"카테고리": "기술", "항목": "EDR", "내용": "엔드포인트 위협 탐지 및 대응"},
        {"카테고리": "기술", "항목": "Zero Trust", "내용": "제로트러스트 보안 아키텍처"},
        {"카테고리": "기술", "항목": "AI 보안", "내용": "머신러닝 기반 위협 분석"},
        {"카테고리": "기술", "항목": "클라우드", "내용": "AWS, Azure, GCP, 멀티클라우드"},
        {"카테고리": "기술", "항목": "컨테이너", "내용": "Docker, Kubernetes 보안"},
        {"카테고리": "기술", "항목": "DevSecOps", "내용": "CI/CD 파이프라인 보안"},
        {"카테고리": "기술", "항목": "API 보안", "내용": "API Gateway, OAuth 2.0"},
        {"카테고리": "기술", "항목": "모바일 보안", "내용": "MDM, MAM, 앱 보안"},
        {"카테고리": "기술", "항목": "IoT 보안", "내용": "IoT 디바이스 인증 및 암호화"},
        {"카테고리": "기술", "항목": "블록체인", "내용": "블록체인 기반 인증 시스템"},
        {"카테고리": "기술", "항목": "양자암호", "내용": "차세대 양자암호 기술 연구"},
        {"카테고리": "기술", "항목": "5G 보안", "내용": "5G 네트워크 보안 솔루션"},
        {"카테고리": "기술", "항목": "SD-WAN", "내용": "소프트웨어 정의 WAN 보안"},
    ]
    data.extend(tech_stack)

    # 9. 수상 실적 (10개)
    awards = []
    for i in range(1, 11):
        year = 2015 + i
        awards.append({
            "카테고리": "수상",
            "항목": f"{year}년 수상",
            "내용": f"{year}년 대한민국 보안 대상 수상 - VPN 부문 최우수상"
        })
    data.extend(awards)

    return data

def save_to_csv(data, filename='company_info_data.csv'):
    """CSV 파일로 저장"""
    fieldnames = ['카테고리', '항목', '내용']

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"✅ {len(data)}개의 회사 소개 데이터 생성 완료: {filename}")

if __name__ == "__main__":
    print("=" * 50)
    print("퓨쳐시스템 회사 소개 데이터 200개 생성")
    print("=" * 50)

    data = generate_company_data()
    save_to_csv(data)

    print(f"\n총 {len(data)}개 항목")
    print("\n카테고리별 분포:")
    categories = {}
    for item in data:
        cat = item['카테고리']
        categories[cat] = categories.get(cat, 0) + 1

    for cat, count in categories.items():
        print(f"- {cat}: {count}개")
