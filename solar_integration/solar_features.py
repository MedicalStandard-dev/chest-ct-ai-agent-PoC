# solar_integration/solar_features.py
"""
Upstage Solar 추가 기능들

1. Groundedness Check - 환각 검증
2. 한국어 번역 - 리포트 한글화
3. Prior 요약 - 과거 리포트 요약
4. Q&A - 의사 질의응답
"""
from typing import Optional, Dict, List, Tuple
import httpx
import re
from dataclasses import dataclass, field

from config.settings import settings
from utils.logger import logger


@dataclass
class GroundednessResult:
    """Groundedness 검증 결과"""
    is_grounded: bool
    confidence: float
    ungrounded_claims: List[str] = field(default_factory=list)
    explanation: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "is_grounded": self.is_grounded,
            "confidence": round(self.confidence, 3),
            "ungrounded_claims": self.ungrounded_claims,
            "explanation": self.explanation
        }


@dataclass
class TranslationResult:
    """번역 결과"""
    original_text: str
    translated_text: str
    source_lang: str = "en"
    target_lang: str = "ko"
    
    def to_dict(self) -> Dict:
        return {
            "original": self.original_text,
            "translated": self.translated_text,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang
        }


@dataclass
class PriorSummary:
    """Prior 요약 결과"""
    summary_text: str
    prior_count: int
    key_changes: List[str] = field(default_factory=list)
    recommendation: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "summary": self.summary_text,
            "prior_count": self.prior_count,
            "key_changes": self.key_changes,
            "recommendation": self.recommendation
        }


@dataclass
class QAResponse:
    """Q&A 응답"""
    question: str
    answer: str
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "confidence": round(self.confidence, 3)
        }


class SolarFeatures:
    """
    Upstage Solar 추가 기능 통합 클래스
    
    - Groundedness Check: Solar로 환각 검증
    - Translation: 한국어 번역
    - Prior Summary: 과거 리포트 요약
    - Q&A: 의사 질의응답
    """
    
    GROUNDEDNESS_PROMPT = """의료 AI 리포트 검증.

규칙: 리포트의 결절 정보(개수, 크기, 위치, confidence)가 context와 일치하면 true.
표준 문구, 경고문, absent findings는 모두 허용.

CONTEXT:
{context}

CLAIM:
{claim}

바로 JSON만 출력:
{{"is_grounded": true, "confidence": 0.95, "explanation": "결절 정보 일치"}}"""

    TRANSLATION_PROMPT = """Translate the following medical report from English to Korean.
Keep medical terminology accurate. Maintain the same structure and formatting.

English Report:
{text}

Korean Translation:"""

    PRIOR_SUMMARY_PROMPT = """다음 과거 의료 리포트들을 영상의학과 의사를 위해 요약하세요.
중점: 병변 변화, 크기 추세, 임상적 의의.
반드시 한국어로 작성하세요.

과거 리포트:
{priors}

다음 형식으로 간결하게 요약:
- 소견 타임라인
- 주요 변화
- 임상 권고 (해당 시)

요약:"""

    QA_PROMPT = """당신은 영상의학과 AI 어시스턴트입니다. 제공된 AI 분석 결과만을 바탕으로 의사의 질문에 답변하세요.
진단이나 치료 권고는 하지 마세요. AI 소견만 설명하세요.
반드시 한국어로 답변하세요.

AI 분석 결과:
{context}

의사 질문: {question}

답변 (간결하고 사실적으로, 한국어로):"""

    def __init__(self):
        self.api_key = settings.solar_api_key
        self.endpoint = settings.solar_api_endpoint
        self.model = getattr(settings, 'solar_report_model', 'upstage/solar-pro-3:free')
        self.use_mock = not settings.should_use_real_solar
        
        if self.use_mock:
            logger.warning("SolarFeatures: Using MOCK mode")
        else:
            logger.info("SolarFeatures: Using REAL Solar API")
    
    async def _call_solar(self, prompt: str, max_tokens: int = 1000) -> str:
        """Solar API 호출"""
        if self.use_mock:
            return self._mock_response(prompt)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.endpoint}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            # 응답 구조 확인
            if not result.get("choices"):
                logger.warning("Solar API returned no choices: {}", result)
                return ""
            
            content = result["choices"][0].get("message", {}).get("content", "")
            if not content:
                logger.warning("Solar API returned empty content. Full response: {}", result)
            
            return content.strip()

    @staticmethod
    def _extract_json_block(text: str) -> str:
        """응답에서 JSON 블록 추출 (없으면 원문 반환)"""
        match = re.search(r'\{.*\}', text, flags=re.DOTALL)
        return match.group(0) if match else text
    
    def _mock_response(self, prompt: str) -> str:
        """Mock 응답 (테스트용)"""
        if "fact-checker" in prompt.lower():
            return '{"is_grounded": true, "confidence": 0.95, "ungrounded_parts": [], "explanation": "All claims are supported by the AI detection results."}'
        elif "translate" in prompt.lower():
            return "이 리포트는 AI 분석 결과입니다. 의사 확인이 필요합니다."
        elif "summarize" in prompt.lower():
            return "Prior Summary: 과거 검사 기록이 없습니다."
        elif "physician" in prompt.lower():
            return "AI 분석 결과에 따르면, 검출된 결절은 confidence 점수가 높습니다. 추가 평가가 권장됩니다."
        return "Mock response"
    
    # ========== 1. Groundedness Check ==========
    
    async def check_groundedness(
        self,
        claim: str,
        context: str
    ) -> GroundednessResult:
        """
        Solar로 환각 검증
        
        Args:
            claim: 검증할 텍스트 (리포트 내용)
            context: 근거 데이터 (AI 분석 결과)
            
        Returns:
            GroundednessResult
        """
        prompt = self.GROUNDEDNESS_PROMPT.format(
            context=context,
            claim=claim
        )
        
        try:
            response = await self._call_solar(prompt, max_tokens=2500)
            
            # Parse JSON response
            import json
            # Clean response
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r'^```json?\s*', '', response)
                response = re.sub(r'\s*```$', '', response)
            if not response:
                raise ValueError("Empty response from Solar")

            response_json = self._extract_json_block(response)
            data = json.loads(response_json)
            
            return GroundednessResult(
                is_grounded=data.get("is_grounded", True),
                confidence=data.get("confidence", 0.5),
                ungrounded_claims=data.get("ungrounded_parts", []),
                explanation=data.get("explanation", "")
            )
            
        except Exception as e:
            logger.error(f"Groundedness check failed: {e}")
            if "response" in locals():
                logger.warning(
                    "Groundedness raw response (truncated): {}",
                    response[:1000]
                )
            # Fail-safe: mark as not grounded when check fails
            return GroundednessResult(
                is_grounded=False,
                confidence=0.0,
                explanation=f"Check failed: {str(e)}"
            )
    
    # ========== 2. 한국어 번역 ==========
    
    async def translate_to_korean(self, text: str) -> TranslationResult:
        """
        리포트를 한국어로 번역
        
        Args:
            text: 영어 리포트 텍스트
            
        Returns:
            TranslationResult
        """
        prompt = self.TRANSLATION_PROMPT.format(text=text)
        
        try:
            translated = await self._call_solar(prompt, max_tokens=2000)
            
            return TranslationResult(
                original_text=text,
                translated_text=translated,
                source_lang="en",
                target_lang="ko"
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return TranslationResult(
                original_text=text,
                translated_text=f"[번역 실패: {str(e)}]",
                source_lang="en",
                target_lang="ko"
            )
    
    # ========== 3. Prior Report 요약 ==========
    
    async def summarize_priors(
        self,
        prior_reports: List[Dict]
    ) -> PriorSummary:
        """
        과거 리포트들을 요약
        
        Args:
            prior_reports: 과거 리포트 목록 [{"date": ..., "findings": ...}, ...]
            
        Returns:
            PriorSummary
        """
        if not prior_reports:
            return PriorSummary(
                summary_text="No prior studies available for comparison.",
                prior_count=0,
                key_changes=[],
                recommendation="Initial study - no prior comparison possible."
            )
        
        # Format priors for prompt
        priors_text = ""
        for i, pr in enumerate(prior_reports, 1):
            date = pr.get("date", "Unknown date")
            findings = pr.get("findings", pr.get("text", "No findings recorded"))
            priors_text += f"\n[Study {i}] Date: {date}\n{findings}\n"
        
        prompt = self.PRIOR_SUMMARY_PROMPT.format(priors=priors_text)
        
        try:
            summary = await self._call_solar(prompt, max_tokens=800)
            
            # Extract key changes (simple parsing)
            key_changes = []
            for line in summary.split('\n'):
                if line.strip().startswith('-'):
                    key_changes.append(line.strip()[1:].strip())
            
            return PriorSummary(
                summary_text=summary,
                prior_count=len(prior_reports),
                key_changes=key_changes[:5],  # Top 5 changes
                recommendation=""
            )
            
        except Exception as e:
            logger.error(f"Prior summary failed: {e}")
            return PriorSummary(
                summary_text=f"[요약 실패: {str(e)}]",
                prior_count=len(prior_reports),
                key_changes=[],
                recommendation=""
            )
    
    # ========== 4. Q&A 인터페이스 ==========
    
    async def answer_question(
        self,
        question: str,
        ai_results: Dict,
        prior_data: Dict = None
    ) -> QAResponse:
        """
        의사 질문에 답변
        
        Args:
            question: 의사의 질문
            ai_results: AI 분석 결과 (context)
            prior_data: 과거 검사 데이터 (optional)
            
        Returns:
            QAResponse
        """
        # Format context (prior 포함)
        context = self._format_ai_results_for_qa(ai_results, prior_data)
        
        prompt = self.QA_PROMPT.format(
            context=context,
            question=question
        )
        
        try:
            answer = await self._call_solar(prompt, max_tokens=1000)
            
            # Extract sources mentioned
            sources = []
            if "nodule" in answer.lower():
                sources.append("Nodule Detection")
            if "confidence" in answer.lower():
                sources.append("AI Confidence Scores")
            if "lung" in answer.lower():
                sources.append("Lung Segmentation")
            
            return QAResponse(
                question=question,
                answer=answer,
                sources=sources or ["AI Analysis Results"],
                confidence=0.85
            )
            
        except Exception as e:
            logger.error(f"Q&A failed: {e}")
            return QAResponse(
                question=question,
                answer=f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}",
                sources=[],
                confidence=0.0
            )
    
    def _format_ai_results_for_qa(self, ai_results: Dict, prior_data: Dict = None) -> str:
        """AI 결과를 Q&A용 컨텍스트로 포맷 (prior 포함)"""
        lines = []
        
        # Nodules
        nodules = ai_results.get("nodules", [])
        if nodules:
            lines.append(f"[현재 검사] 검출된 결절 후보: {len(nodules)}개")
            for n in nodules[:5]:  # Top 5
                loc = n.get("location_code", "UNK")
                diam = n.get("diameter_mm", 0)
                conf = n.get("confidence", 0)
                lines.append(f"  - 위치: {loc}, 직경: {diam:.1f}mm, 신뢰도: {conf:.2f}")
        
        # Key flags
        key_flags = ai_results.get("key_flags", {})
        if key_flags:
            lines.append(f"\n[주요 소견]")
            lines.append(f"  - 총 후보 수: {key_flags.get('nodule_candidates', 0)}")
            lines.append(f"  - 고신뢰도 소견: {key_flags.get('high_confidence_findings', 0)}")
        
        # Quality
        quality = ai_results.get("quality", {})
        if quality:
            lines.append(f"\n[영상 품질]")
            lines.append(f"  - 슬라이스 두께: {quality.get('slice_thickness_mm', 'N/A')}mm")
        
        # Prior data (과거 검사)
        if prior_data:
            lines.append(f"\n[과거 검사 기록]")
            lines.append(f"  - 검사일: {prior_data.get('study_date', 'Unknown')}")
            if prior_data.get('nodule_diameter_mm'):
                lines.append(f"  - 과거 결절 크기: {prior_data.get('nodule_diameter_mm'):.1f}mm")
                lines.append(f"  - 과거 결절 위치: {prior_data.get('nodule_location', 'Unknown')}")
                
                # 크기 변화 계산
                current_nodules = ai_results.get("nodules", [])
                if current_nodules:
                    current_size = current_nodules[0].get("diameter_mm", 0)
                    prior_size = prior_data.get('nodule_diameter_mm', 0)
                    if prior_size > 0:
                        change_pct = ((current_size - prior_size) / prior_size) * 100
                        lines.append(f"  - 크기 변화: {change_pct:+.1f}% ({'증가' if change_pct > 0 else '감소' if change_pct < 0 else '안정'})")
        else:
            lines.append(f"\n[과거 검사 기록] 없음")
        
        return "\n".join(lines) if lines else "분석 결과가 없습니다."
    
    # ========== 통합 기능 ==========
    
    async def enhance_report(
        self,
        report_text: str,
        ai_results: Dict,
        include_korean: bool = True,
        check_groundedness: bool = True
    ) -> Dict:
        """
        리포트 강화 (한국어 번역 + Groundedness Check)
        
        Returns:
            {
                "english": original report,
                "korean": translated report,
                "groundedness": check result,
                "solar_contributions": list of contributions
            }
        """
        result = {
            "english": report_text,
            "korean": None,
            "groundedness": None,
            "solar_contributions": []
        }
        
        # 1. Groundedness Check
        if check_groundedness:
            context = self._format_ai_results_for_qa(ai_results)
            groundedness = await self.check_groundedness(report_text, context)
            result["groundedness"] = groundedness.to_dict()
            result["solar_contributions"].append("Groundedness Check by Solar Pro 3")
        
        # 2. Korean Translation
        if include_korean:
            translation = await self.translate_to_korean(report_text)
            result["korean"] = translation.translated_text
            result["solar_contributions"].append("Korean Translation by Solar Pro 3")
        
        return result


# Singleton instance
_solar_features: Optional[SolarFeatures] = None

def get_solar_features() -> SolarFeatures:
    """Get or create SolarFeatures instance"""
    global _solar_features
    if _solar_features is None:
        _solar_features = SolarFeatures()
    return _solar_features
