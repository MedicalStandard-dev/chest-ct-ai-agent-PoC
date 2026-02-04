# solar_integration/rag_system.py
"""
RAG system for prior study retrieval
Solar Embedding + ChromaDB + Patient isolation
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import re

import chromadb
from chromadb.config import Settings as ChromaSettings

from solar_integration.embeddings import SolarEmbeddingClient
from solar_integration.tracking import LesionTracker
from api.schemas import StructuredAIResult
from config.settings import settings
from utils.logger import logger


class MedicalRAGSystem:
    """
    RAG for medical report history
    - Patient isolation 강제
    - Lesion tracking 우선, semantic search 보조
    """
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        use_mock_embedding: bool = False
    ):
        """
        Args:
            db_path: ChromaDB 저장 경로
            use_mock_embedding: Mock embedding 강제 사용
        """
        self.db_path = db_path or Path(getattr(settings, 'chroma_db_path', './data/chroma_db'))
        
        # Initialize embedding client
        self.embedding_client = SolarEmbeddingClient(use_mock=use_mock_embedding)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Collections
        self.reports_collection = self.chroma_client.get_or_create_collection(
            name="radiology_reports",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.ai_results_collection = self.chroma_client.get_or_create_collection(
            name="ai_analysis_results",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Lesion tracker
        self.tracker = LesionTracker()
        
        logger.info(f"Initialized MedicalRAGSystem with {self.embedding_client.get_model_info()}")
    
    async def store_report(
        self,
        patient_id: str,
        study_uid: str,
        study_date: str,
        report_text: str,
        ai_result: StructuredAIResult,
        metadata: Optional[Dict] = None
    ):
        """
        리포트 및 AI 결과 저장
        
        Args:
            patient_id: 환자 ID (필수)
            study_uid: Study UID
            study_date: Study date (YYYYMMDD)
            report_text: 판독문 텍스트
            ai_result: 구조화된 AI 결과
            metadata: 추가 메타데이터
        """
        if not patient_id:
            raise ValueError("patient_id is required for RAG storage")
        
        doc_id = f"{patient_id}_{study_uid}"
        normalized_study_date = self._normalize_study_date(study_date)
        now_iso = datetime.now().isoformat()
        
        # Existing 문서가 있으면 created_at 유지 (upsert=update 동작 명시)
        existing_created_at = now_iso
        existing = self.reports_collection.get(ids=[doc_id])
        if existing and existing.get("ids"):
            existing_meta = (existing.get("metadatas") or [{}])[0] or {}
            existing_created_at = (
                existing_meta.get("created_at")
                or existing_meta.get("timestamp")
                or now_iso
            )
        
        # Prepare metadata
        largest_nodule = max(ai_result.nodules, key=lambda n: n.diameter_mm, default=None)
        doc_metadata = {
            "patient_id": patient_id,
            "study_uid": study_uid,
            "study_date": normalized_study_date,
            "study_date_display": study_date,
            "timestamp": now_iso,  # backward compatibility
            "created_at": existing_created_at,
            "updated_at": now_iso,
            "is_updated": bool(existing and existing.get("ids")),
            "num_nodules": len(ai_result.nodules),
            "model_version": ai_result.versioning.model_version,
            "pipeline_version": ai_result.versioning.pipeline_version,
            "max_nodule_diameter_mm": float(largest_nodule.diameter_mm) if largest_nodule else 0.0,
            "max_nodule_confidence": float(largest_nodule.confidence) if largest_nodule else 0.0,
            "nodule_location": (largest_nodule.location_code if largest_nodule else "UNK"),
            "nodule_diameter_mm": float(largest_nodule.diameter_mm) if largest_nodule else 0.0
        }
        if metadata:
            doc_metadata.update(metadata)
        
        # Embed report text
        report_embedding = await self.embedding_client.embed_single(report_text)
        
        # Store in reports collection
        self.reports_collection.upsert(
            ids=[doc_id],
            embeddings=[report_embedding],
            documents=[report_text],
            metadatas=[doc_metadata]
        )
        
        # Store AI results as JSON
        ai_results_text = json.dumps(ai_result.model_dump(mode="json"), indent=2, default=str)
        ai_embedding = await self.embedding_client.embed_single(ai_results_text)
        
        self.ai_results_collection.upsert(
            ids=[doc_id],
            embeddings=[ai_embedding],
            documents=[ai_results_text],
            metadatas=[doc_metadata]
        )
        
        logger.info(
            f"Stored report and AI results for {doc_id} "
            f"(study_date={normalized_study_date}, updated={doc_metadata['is_updated']})"
        )
    
    def retrieve_patient_history(
        self,
        patient_id: str,
        max_results: int = 10
    ) -> List[Dict]:
        """
        환자의 모든 과거 검사 조회
        
        CRITICAL: patient_id 필수 (patient isolation)
        """
        if not patient_id:
            raise ValueError("patient_id is required for history retrieval")
        
        results = self.reports_collection.get(
            where={"patient_id": patient_id},
            limit=max_results
        )
        
        history = []
        if results and results['ids']:
            for i in range(len(results['ids'])):
                history.append({
                    "study_uid": results['metadatas'][i]['study_uid'],
                    "study_date": results['metadatas'][i]['study_date'],
                    "report_text": results['documents'][i],
                    "metadata": results['metadatas'][i]
                })
        
        # Sort by date (newest first)
        history.sort(
            key=lambda x: self._normalize_study_date(x.get("study_date", "")),
            reverse=True
        )
        
        logger.info(f"Retrieved {len(history)} studies for patient {patient_id}")
        return history
    
    def retrieve_most_recent_prior(
        self,
        patient_id: str,
        current_study_date: str
    ) -> Optional[Dict]:
        """
        가장 최근 prior study 조회
        
        Args:
            patient_id: 환자 ID (필수)
            current_study_date: 현재 검사 날짜 (YYYYMMDD)
        """
        if not patient_id:
            raise ValueError("patient_id is required for prior retrieval")
        
        history = self.retrieve_patient_history(patient_id, max_results=20)
        
        # Filter for studies before current date
        prior_studies = [
            h for h in history
            if self._normalize_study_date(h.get("study_date", "")) < self._normalize_study_date(current_study_date)
        ]
        
        if prior_studies:
            most_recent = prior_studies[0]  # Already sorted
            logger.info(f"Found prior study from {most_recent['study_date']}")
            return most_recent
        
        logger.info("No prior study found")
        return None
    
    def retrieve_prior_ai_result(
        self,
        patient_id: str,
        study_uid: str
    ) -> Optional[StructuredAIResult]:
        """
        과거 검사의 AI 결과 조회
        
        Lesion tracking에 사용
        """
        if not patient_id:
            return None
        
        doc_id = f"{patient_id}_{study_uid}"
        
        results = self.ai_results_collection.get(
            ids=[doc_id]
        )
        
        if results and results['documents']:
            ai_result_dict = json.loads(results['documents'][0])
            # Convert back to StructuredAIResult
            # (실제로는 Pydantic parse_obj 사용)
            logger.info(f"Retrieved AI result for {doc_id}")
            return ai_result_dict  # Simplified
        
        return None
    
    async def semantic_search(
        self,
        query_text: str,
        patient_id: str,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Semantic search (patient isolation 강제)
        
        Args:
            query_text: 검색 쿼리
            patient_id: 환자 ID (필수)
            n_results: 결과 개수
        """
        if not patient_id:
            raise ValueError("patient_id is required for semantic search (patient isolation)")
        
        # Embed query
        query_embedding = await self.embedding_client.embed_single(query_text)
        
        # Search with patient filter
        results = self.reports_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"patient_id": patient_id}
        )
        
        similar_cases = []
        if results and results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                similar_cases.append({
                    "study_uid": results['metadatas'][0][i]['study_uid'],
                    "study_date": results['metadatas'][0][i]['study_date'],
                    "report_text": results['documents'][0][i],
                    "similarity_score": 1 - results['distances'][0][i],
                    "metadata": results['metadatas'][0][i]
                })
        
        logger.info(f"Found {len(similar_cases)} similar cases for patient {patient_id}")
        return similar_cases
    
    async def compare_with_prior(
        self,
        current_result: StructuredAIResult,
        patient_id: str,
        current_study_date: str
    ) -> Optional[Dict]:
        """
        Prior study와 비교 (tracking + semantic search)
        
        Returns:
            {
                "prior_study": {...},
                "lesion_tracking": {...},
                "comparison_text": str
            }
        """
        if not patient_id:
            logger.warning("No patient_id: skipping prior comparison")
            return None
        
        # 1. Retrieve most recent prior
        prior = self.retrieve_most_recent_prior(patient_id, current_study_date)
        
        if not prior:
            logger.info("No prior study available for comparison")
            return None
        
        # 2. Load prior AI result
        prior_ai_result_dict = self.retrieve_prior_ai_result(
            patient_id,
            prior['study_uid']
        )
        
        if not prior_ai_result_dict:
            logger.warning("Prior AI result not found - comparison limited")
            # prior_data는 metadata에서 가져옴
            prior_data = {
                "study_date": prior.get("study_date"),
                "study_uid": prior.get("study_uid"),
                "nodule_diameter_mm": prior.get("metadata", {}).get("nodule_diameter_mm"),
                "nodule_location": prior.get("metadata", {}).get("nodule_location")
            }
            return {
                "prior_study": prior,
                "prior_data": prior_data,  # 추가!
                "comparison_text": "Prior study available but detailed comparison unavailable.",
                "priors_count": 1
            }
        
        # 3. Lesion tracking (규칙 기반)
        try:
            # Convert dict to StructuredAIResult
            # (실제로는 Pydantic validation 필요)
            # Simplified: assume compatible structure
            from api.schemas import StructuredAIResult as SAR
            prior_result = SAR(**prior_ai_result_dict)
            
            tracking_result = self.tracker.compare_studies(
                current_result,
                prior_result
            )
            
            comparison_text = self.tracker.generate_comparison_text(tracking_result)
            
            # prior_data 추출 (PRIOR COMPARISON 테이블용)
            prior_data = {
                "study_date": prior.get("study_date"),
                "study_uid": prior.get("study_uid"),
                "nodule_diameter_mm": prior.get("metadata", {}).get("nodule_diameter_mm"),
                "nodule_location": prior.get("metadata", {}).get("nodule_location")
            }
            
            return {
                "prior_study": prior,
                "prior_data": prior_data,  # 추가
                "lesion_tracking": tracking_result,
                "comparison_text": comparison_text,
                "significant_changes": self.tracker.get_significant_changes(tracking_result),
                "priors_count": 1
            }
            
        except Exception as e:
            logger.error(f"Lesion tracking failed: {e}")
            # 에러 시에도 prior_data 제공
            prior_data = {
                "study_date": prior.get("study_date"),
                "study_uid": prior.get("study_uid"),
                "nodule_diameter_mm": prior.get("metadata", {}).get("nodule_diameter_mm"),
                "nodule_location": prior.get("metadata", {}).get("nodule_location")
            }
            return {
                "prior_study": prior,
                "prior_data": prior_data,
                "comparison_text": f"Prior study available (tracking error: {e})",
                "priors_count": 1
            }
    
    def get_rag_info(self) -> Dict:
        """RAG 시스템 정보"""
        return {
            "embedding_model": self.embedding_client.get_model_info(),
            "vector_db": "ChromaDB",
            "patient_isolation": True,
            "lesion_tracking": True
        }

    def _normalize_study_date(self, study_date: str) -> str:
        """
        다양한 study_date 형식을 YYYYMMDD 문자열로 정규화
        예: 2026-02-04, 2026/02/04, 20260204
        """
        if not study_date:
            return ""
        
        raw = str(study_date).strip()
        for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(raw, fmt).strftime("%Y%m%d")
            except ValueError:
                pass
        
        # fallback: 숫자만 추출 후 8자리면 날짜로 간주
        digits = re.sub(r"[^0-9]", "", raw)
        if len(digits) >= 8:
            return digits[:8]
        
        return raw
