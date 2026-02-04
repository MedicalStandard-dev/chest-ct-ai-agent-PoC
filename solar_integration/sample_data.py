# solar_integration/sample_data.py
"""
RAG 데모용 샘플 과거 환자 데이터
"""
from typing import List, Dict
from datetime import datetime, timedelta

# 샘플 과거 리포트 데이터 (10개)
SAMPLE_PRIOR_REPORTS: List[Dict] = [
    # LIDC-IDRI-0001: 6개월 전 리포트 (결절 크기 증가 시뮬레이션)
    {
        "patient_id": "LIDC-IDRI-0001",
        "study_uid": "prior-0001-2025-08",
        "study_date": "2025-08-03",
        "report_text": """CT Chest | 2025-08-03 | LIDC-IDRI-0001
FINDINGS:
- 5.2 mm solid nodule in right upper lobe (RUL), subpleural location
- No other pulmonary nodules identified
- Lungs otherwise clear
IMPRESSION:
- Small pulmonary nodule in RUL, recommend follow-up CT in 6 months
- Lung-RADS 3: Probably benign""",
        "nodule_info": {"location": "RUL", "diameter_mm": 5.2, "confidence": 0.85}
    },
    
    # LIDC-IDRI-0001: 12개월 전 리포트
    {
        "patient_id": "LIDC-IDRI-0001",
        "study_uid": "prior-0001-2025-02",
        "study_date": "2025-02-03",
        "report_text": """CT Chest | 2025-02-03 | LIDC-IDRI-0001
FINDINGS:
- 4.1 mm ground-glass nodule in right upper lobe
- No significant mediastinal lymphadenopathy
IMPRESSION:
- Small GGN in RUL, likely benign
- Recommend follow-up in 6 months""",
        "nodule_info": {"location": "RUL", "diameter_mm": 4.1, "confidence": 0.72}
    },
    
    # LIDC-IDRI-0021: 이전 스캔
    {
        "patient_id": "LIDC-IDRI-0021",
        "study_uid": "prior-0021-2025-09",
        "study_date": "2025-09-15",
        "report_text": """CT Chest | 2025-09-15 | LIDC-IDRI-0021
FINDINGS:
- 8.3 mm part-solid nodule in left lower lobe (LLL)
- 3.2 mm solid nodule in right middle lobe (RML)
IMPRESSION:
- Part-solid nodule in LLL requires close follow-up
- Small nodule in RML, likely benign
- Lung-RADS 4A: Suspicious""",
        "nodule_info": {"location": "LLL", "diameter_mm": 8.3, "confidence": 0.91}
    },
    
    # LIDC-IDRI-0041: 이전 스캔
    {
        "patient_id": "LIDC-IDRI-0041",
        "study_uid": "prior-0041-2025-07",
        "study_date": "2025-07-22",
        "report_text": """CT Chest | 2025-07-22 | LIDC-IDRI-0041
FINDINGS:
- Multiple pulmonary nodules identified
- Largest: 12.5 mm in right upper lobe
- 6.8 mm nodule in left upper lobe
- 4.2 mm nodule in right lower lobe
IMPRESSION:
- Multiple pulmonary nodules, recommend PET-CT
- Lung-RADS 4B: Suspicious""",
        "nodule_info": {"location": "RUL", "diameter_mm": 12.5, "confidence": 0.95}
    },
    
    # LIDC-IDRI-0061: 이전 스캔 (안정적)
    {
        "patient_id": "LIDC-IDRI-0061",
        "study_uid": "prior-0061-2025-06",
        "study_date": "2025-06-10",
        "report_text": """CT Chest | 2025-06-10 | LIDC-IDRI-0061
FINDINGS:
- 6.0 mm calcified granuloma in right middle lobe
- No new nodules
IMPRESSION:
- Stable calcified granuloma, benign
- No follow-up needed
- Lung-RADS 1: Negative""",
        "nodule_info": {"location": "RML", "diameter_mm": 6.0, "confidence": 0.88}
    },
    
    # LIDC-IDRI-0081: 이전 스캔
    {
        "patient_id": "LIDC-IDRI-0081",
        "study_uid": "prior-0081-2025-05",
        "study_date": "2025-05-18",
        "report_text": """CT Chest | 2025-05-18 | LIDC-IDRI-0081
FINDINGS:
- 9.1 mm spiculated nodule in left upper lobe
- Suspicious morphology
IMPRESSION:
- Highly suspicious nodule in LUL
- Recommend tissue sampling
- Lung-RADS 4B: Suspicious""",
        "nodule_info": {"location": "LUL", "diameter_mm": 9.1, "confidence": 0.94}
    },
    
    # LIDC-IDRI-0101: 이전 스캔
    {
        "patient_id": "LIDC-IDRI-0101",
        "study_uid": "prior-0101-2025-04",
        "study_date": "2025-04-05",
        "report_text": """CT Chest | 2025-04-05 | LIDC-IDRI-0101
FINDINGS:
- No pulmonary nodules identified
- Clear lung parenchyma
- Normal mediastinum
IMPRESSION:
- No evidence of pulmonary nodules
- Lung-RADS 1: Negative""",
        "nodule_info": None
    },
    
    # LIDC-IDRI-0121: 이전 스캔
    {
        "patient_id": "LIDC-IDRI-0121",
        "study_uid": "prior-0121-2025-03",
        "study_date": "2025-03-12",
        "report_text": """CT Chest | 2025-03-12 | LIDC-IDRI-0121
FINDINGS:
- 7.5 mm solid nodule in right lower lobe
- Smooth margins, no spiculation
IMPRESSION:
- Indeterminate nodule in RLL
- Recommend 3-month follow-up
- Lung-RADS 3: Probably benign""",
        "nodule_info": {"location": "RLL", "diameter_mm": 7.5, "confidence": 0.82}
    },
    
    # LIDC-IDRI-0141: 이전 스캔 (2개 시점)
    {
        "patient_id": "LIDC-IDRI-0141",
        "study_uid": "prior-0141-2025-08",
        "study_date": "2025-08-20",
        "report_text": """CT Chest | 2025-08-20 | LIDC-IDRI-0141
FINDINGS:
- 5.8 mm nodule in lingula, stable compared to prior
- No new nodules
IMPRESSION:
- Stable nodule in lingula over 6 months
- Continue annual surveillance
- Lung-RADS 2: Benign""",
        "nodule_info": {"location": "LINGULA", "diameter_mm": 5.8, "confidence": 0.79}
    },
    
    # LIDC-IDRI-0141: 더 이전 스캔
    {
        "patient_id": "LIDC-IDRI-0141",
        "study_uid": "prior-0141-2025-02",
        "study_date": "2025-02-15",
        "report_text": """CT Chest | 2025-02-15 | LIDC-IDRI-0141
FINDINGS:
- 5.6 mm nodule in lingula, new finding
IMPRESSION:
- New small nodule in lingula
- Recommend 6-month follow-up
- Lung-RADS 3: Probably benign""",
        "nodule_info": {"location": "LINGULA", "diameter_mm": 5.6, "confidence": 0.75}
    },
]


async def seed_sample_data(rag_system):
    """
    RAG 시스템에 샘플 데이터 삽입
    
    Args:
        rag_system: MedicalRAGSystem instance
    """
    from utils.logger import logger
    
    logger.info("="*60)
    logger.info("SEEDING SAMPLE PRIOR REPORTS FOR RAG DEMO")
    logger.info("="*60)
    
    seeded_count = 0
    
    for report in SAMPLE_PRIOR_REPORTS:
        try:
            # 간단한 저장 (embedding만)
            doc_id = f"{report['patient_id']}_{report['study_uid']}"
            
            # Check if already exists
            existing = rag_system.reports_collection.get(ids=[doc_id])
            if existing and existing['ids']:
                logger.info(f"  Skip (exists): {doc_id}")
                continue
            
            # Embed and store
            embedding = await rag_system.embedding_client.embed_single(report['report_text'])
            
            metadata = {
                "patient_id": report['patient_id'],
                "study_uid": report['study_uid'],
                "study_date": report['study_date'],
                "timestamp": datetime.now().isoformat(),
                "is_sample_data": True
            }
            
            if report.get('nodule_info'):
                metadata['nodule_location'] = report['nodule_info']['location']
                metadata['nodule_diameter_mm'] = report['nodule_info']['diameter_mm']
            
            rag_system.reports_collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[report['report_text']],
                metadatas=[metadata]
            )
            
            seeded_count += 1
            logger.info(f"  ✓ Seeded: {report['patient_id']} ({report['study_date']})")
            
        except Exception as e:
            logger.warning(f"  ✗ Failed to seed {report.get('patient_id')}: {e}")
    
    logger.info(f"Seeded {seeded_count}/{len(SAMPLE_PRIOR_REPORTS)} sample reports")
    logger.info("="*60)
    
    return seeded_count


def get_patient_prior_summary(patient_id: str) -> List[Dict]:
    """특정 환자의 샘플 과거 데이터 반환"""
    return [
        r for r in SAMPLE_PRIOR_REPORTS 
        if r['patient_id'] == patient_id
    ]
