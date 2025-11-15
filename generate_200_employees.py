"""
퓨쳐시스템 200명 직원 데이터 생성
"""
import csv
import random

# 한국 이름 풀
family_names = ['김', '이', '박', '최', '정', '강', '조', '윤', '장', '임', '한', '오', '서', '신', '권', '황', '안', '송', '류', '홍']
given_names = ['민준', '서준', '예준', '도윤', '시우', '주원', '하준', '지호', '지후', '준서',
               '서연', '서윤', '지우', '서현', '민서', '하은', '지민', '수아', '윤서', '지유',
               '은우', '건우', '현우', '선우', '연우', '유준', '정우', '승우', '민재', '시후',
               '채원', '다은', '수빈', '소율', '예은', '지안', '수현', '예린', '채은', '소윤']

departments = [
    'VPN팀', '보안팀', '개발팀', '인프라팀', '클라우드팀', '데이터팀', 'AI팀',
    'DevOps팀', 'QA팀', '기획팀', '영업팀', '마케팅팀', '인사팀', '재무팀',
    '총무팀', '법무팀', '구매팀', '고객지원팀', '프로젝트팀', '연구개발팀'
]

positions = ['사원', '주임', '대리', '과장', '차장', '부장', '이사', '상무']

tasks = [
    'VPN 업무', '네트워크 보안', '서버 관리', '시스템 개발', '클라우드 운영',
    '데이터 분석', 'AI 모델 개발', '인프라 구축', '품질 관리', '사업 기획',
    '영업 지원', '마케팅 전략', '인사 관리', '재무 분석', '총무 업무',
    '법률 검토', '구매 관리', 'CS 대응', '프로젝트 관리', '기술 연구'
]

def generate_phone():
    """010으로 시작하는 전화번호 생성"""
    return f"010-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"

def generate_email(name, dept):
    """이메일 생성"""
    # 이름을 영문으로 변환 (간단히 이니셜 사용)
    initial = name[0]
    domain_map = {
        'VPN팀': 'vpn',
        '보안팀': 'security',
        '개발팀': 'dev',
        '인프라팀': 'infra',
        '클라우드팀': 'cloud'
    }
    domain = domain_map.get(dept, 'future')
    return f"{initial.lower()}{random.randint(100, 999)}@{domain}.co.kr"

def generate_employees():
    """200명 직원 데이터 생성"""
    employees = []

    # 1번: 정원규 (대표이사) - 고정
    employees.append({
        '이름': '정원규',
        '직급': '대표이사',
        '부서': '퓨쳐시스템',
        '전화번호': '010-777-7777',
        '이메일': 'ceo@future.co.kr',
        '입사일': '9999-99-99',
        '담당업무': '전사 경영 총괄'
    })

    # 2번: 염재준 (사원, VPN팀) - 고정
    employees.append({
        '이름': '염재준',
        '직급': '사원',
        '부서': 'VPN팀',
        '전화번호': '010-3839-3418',
        '이메일': 'jjyeom@future.co.kr',
        '입사일': '2022-01-10',
        '담당업무': 'VPN 업무'
    })

    # 3~200번: 랜덤 생성
    used_phones = {'010-777-7777', '010-3839-3418'}

    for i in range(3, 201):
        name = random.choice(family_names) + random.choice(given_names)
        dept = random.choice(departments)
        position = random.choice(positions)
        task = random.choice(tasks)

        # 중복되지 않는 전화번호 생성
        while True:
            phone = generate_phone()
            if phone not in used_phones:
                used_phones.add(phone)
                break

        email = generate_email(name, dept)

        # 입사일 랜덤 생성 (2018-2024)
        year = random.randint(2018, 2024)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        join_date = f"{year}-{month:02d}-{day:02d}"

        employees.append({
            '이름': name,
            '직급': position,
            '부서': dept,
            '전화번호': phone,
            '이메일': email,
            '입사일': join_date,
            '담당업무': task
        })

    return employees

def save_to_csv(employees, filename='company_data.csv'):
    """CSV 파일로 저장"""
    fieldnames = ['이름', '직급', '부서', '전화번호', '이메일', '입사일', '담당업무']

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(employees)

    print(f"✅ {len(employees)}명의 직원 데이터 생성 완료: {filename}")

if __name__ == "__main__":
    print("=" * 50)
    print("퓨쳐시스템 200명 직원 데이터 생성")
    print("=" * 50)

    employees = generate_employees()
    save_to_csv(employees)

    print(f"\n총 {len(employees)}명")
    print(f"- 정원규 (대표이사): 010-777-7777")
    print(f"- 염재준 (사원, VPN팀): 010-3839-3418")
    print(f"- 기타 {len(employees) - 2}명")
