# solar_integration/solar_features.py
"""
Upstage Solar 추가 기능들

1. Groundedness Check - 환각 검증
2. 한국어 번역 - 리포트 한글화
3. Prior 요약 - 과거 리포트 요약
4. Q&A - 의사 질의응답
"""
from typing import Any, Optional, Dict, List, Tuple
import asyncio
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

    WHY_CARD_PROMPT = """당신은 영상의학과 보조 AI입니다.
아래 결절 후보의 수치 근거를 바탕으로 왜 이 후보가 표시되었는지 임상적으로 짧게 설명하세요.
진단 확정 표현 금지. 수치 기반으로만 답하세요. 반드시 한국어로 작성하세요.

CASE CONTEXT:
{context}

NODULE:
{nodule}

다음 JSON만 반환:
{{
  "clinical_reason": "왜 후보인지 (1문장)",
  "confidence_reason": "confidence 해석 (1문장)",
  "risk_hint": "판독 시 확인 포인트 (1문장)"
}}"""

    PRIOR_DELTA_PROMPT = """당신은 영상의학과 비교판독 보조 AI입니다.
현재 검사와 과거 검사 비교 정보를 받아, 임상적으로 의미 있는 변화만 한국어로 요약하세요.
반드시 수치를 포함하세요.

CURRENT:
{current}

PRIOR:
{prior}

다음 JSON만 반환:
{{
  "summary": "2~3문장 요약",
  "change_type": "NEW|STABLE|GROWTH|REDUCTION|UNKNOWN",
  "clinical_note": "판독 시 유의점 1문장"
}}"""

    ACTION_SUGGEST_PROMPT = """당신은 영상의학과 workflow 보조 AI입니다.
아래 근거 데이터만으로 다음 액션을 제안하세요.
치료 처방/최종진단은 금지합니다.
반드시 근거 수치를 포함하세요. 반드시 한국어로 작성하세요.

EVIDENCE:
{evidence}

JSON만 반환:
{{
  "recommendation_level": "monitor|review_now|routine_followup",
  "rationale": "근거 수치 포함 1~2문장",
  "next_step": "실무 액션 1문장"
}}"""

    THRESHOLD_EXPLAIN_PROMPT = """당신은 폐결절 검출 운영 정책 보조 AI입니다.
선택 임계값과 성능 지표를 읽고, 해당 설정이 screening/reporting 목적에 적합한지 한국어로 설명하세요.
반드시 sensitivity와 FP/scan 숫자를 직접 언급하세요.

SELECTED:
{selected}

BEST_SCREENING:
{best_screening}

BEST_REPORTING:
{best_reporting}

MODE: {mode}

한 단락(최대 4문장)으로 답변:"""

    # Concurrency limit for Solar API calls to avoid 429 rate limiting
    _semaphore: Optional[asyncio.Semaphore] = None
    _MAX_CONCURRENT = 1
    _last_call_time: float = 0.0
    _MIN_INTERVAL = 1.0  # seconds between calls

    @classmethod
    def _get_semaphore(cls) -> asyncio.Semaphore:
        if cls._semaphore is None:
            cls._semaphore = asyncio.Semaphore(cls._MAX_CONCURRENT)
        return cls._semaphore

    def __init__(self):
        self.api_key = settings.solar_api_key
        self.endpoint = settings.solar_api_endpoint
        self.model = getattr(settings, 'solar_report_model', 'upstage/solar-pro-3:free')
        self.use_mock = not settings.should_use_real_solar

        if self.use_mock:
            logger.warning("SolarFeatures: Using MOCK mode")
        else:
            logger.info("SolarFeatures: Using REAL Solar API")
    
    async def _call_solar(
        self,
        prompt: str,
        max_tokens: int = 1000,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        """Solar API 호출 (429 exponential backoff + concurrency limit)"""
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
        if response_format is not None:
            payload["response_format"] = response_format

        max_retries = 4
        base_delay = 2.0  # seconds

        import time
        async with self._get_semaphore():
            # 최소 호출 간격 강제
            elapsed = time.monotonic() - SolarFeatures._last_call_time
            if elapsed < SolarFeatures._MIN_INTERVAL:
                await asyncio.sleep(SolarFeatures._MIN_INTERVAL - elapsed)
            SolarFeatures._last_call_time = time.monotonic()

            async with httpx.AsyncClient(timeout=60.0) as client:
                for attempt in range(max_retries):
                    response = await client.post(
                        f"{self.endpoint}/chat/completions",
                        headers=headers,
                        json=payload
                    )

                    if response.status_code == 429:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)  # 2, 4, 8s
                            logger.warning(
                                "Solar API 429 rate-limited, retrying in {:.0f}s (attempt {}/{})",
                                delay, attempt + 1, max_retries
                            )
                            await asyncio.sleep(delay)
                            continue
                        # Last attempt still 429 → raise
                        response.raise_for_status()

                    if response.status_code == 502:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            logger.warning(
                                "Solar API 502 Bad Gateway, retrying in {:.0f}s (attempt {}/{})",
                                delay, attempt + 1, max_retries
                            )
                            await asyncio.sleep(delay)
                            continue
                        response.raise_for_status()

                    response.raise_for_status()
                    break

                result = response.json()

                # 응답 구조 확인
                if not result.get("choices"):
                    logger.warning("Solar API returned no choices: {}", result)
                    return ""

                choice0 = result["choices"][0]
                content = choice0.get("message", {}).get("content", "")
                if not content:
                    finish_reason = choice0.get("finish_reason", "unknown")
                    logger.warning(
                        "Solar API returned empty content (finish_reason={}). Full response: {}",
                        finish_reason,
                        result
                    )

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

    @staticmethod
    def _safe_json_parse(text: str) -> Dict:
        import json
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Empty response content")
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```json?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = SolarFeatures._extract_json_block(cleaned)
        return json.loads(cleaned)

    async def _call_solar_json(
        self,
        prompt: str,
        schema_name: str,
        json_schema: Dict[str, Any],
        max_tokens: int = 1200,
        retries: int = 2
    ) -> Dict[str, Any]:
        """
        Structured JSON call with:
        1) response_format=json_schema
        2) retry with JSON-only repair prompt on parse failure
        """
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": json_schema
            }
        }
        errors: List[str] = []

        for attempt in range(retries + 1):
            use_schema = attempt == 0
            attempt_prompt = prompt
            if attempt > 0:
                # Wait before retry to avoid hammering the API
                await asyncio.sleep(1.0 * attempt)
                attempt_prompt = (
                    f"{prompt}\n\n"
                    "IMPORTANT: Return ONLY a valid JSON object. "
                    "No markdown, no explanation, no extra keys."
                )
            try:
                text = await self._call_solar(
                    attempt_prompt,
                    max_tokens=max_tokens + (attempt * 200),
                    response_format=response_format if use_schema else None
                )
                data = self._safe_json_parse(text)
                data["_meta"] = {
                    "generated_by_llm": True,
                    "generator": "Solar Pro 3",
                    "attempt": attempt + 1,
                    "used_json_schema": use_schema,
                }
                return data
            except Exception as e:
                errors.append(str(e))
                logger.warning(
                    "_call_solar_json attempt {} failed (schema={}): {}",
                    attempt + 1,
                    use_schema,
                    e
                )

        raise ValueError("JSON generation failed: " + " | ".join(errors[-3:]))
    
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

            # Structured sources: nodule id / slice / confidence / prior date
            sources = []
            nodules = ai_results.get("nodules", []) if isinstance(ai_results, dict) else []
            for n in nodules[:3]:
                nid = n.get("id", "N?")
                conf = float(n.get("confidence", 0.0))
                srange = n.get("evidence", {}).get("slice_range", [0, 0])
                sources.append(f"{nid}|conf={conf:.2f}|slice={srange[0]}-{srange[1]}")
            if prior_data and prior_data.get("study_date"):
                sources.append(f"prior_date={prior_data.get('study_date')}")
            if not sources:
                sources.append("AI_ANALYSIS_NO_NODULE_SOURCE")
            
            return QAResponse(
                question=question,
                answer=answer,
                sources=sources,
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

    async def explain_nodule_why(self, nodule: Dict, context: Dict, llm_only: bool = False) -> Dict:
        """Generate concise rationale card for a nodule candidate."""
        if self.use_mock:
            conf = float(nodule.get("confidence", 0.0))
            size = float(nodule.get("diameter_mm", 0.0))
            loc = nodule.get("location_code", "UNK")
            return {
                "clinical_reason": f"{loc} 위치의 {size:.1f}mm 후보가 heatmap에서 연결 성분으로 검출되었습니다.",
                "confidence_reason": f"confidence {conf:.2f}는 현재 운영 임계값 대비 {'상회' if conf >= 0.15 else '미달'} 수준입니다.",
                "risk_hint": "주변 혈관/흉막 인접 여부를 오버레이와 함께 확인하세요.",
                "generated_by_llm": False,
                "generator": "MOCK",
                "status": "mock"
            }

        prompt = self.WHY_CARD_PROMPT.format(
            context=str(context),
            nodule=str(nodule)
        )
        try:
            schema = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "clinical_reason": {"type": "string"},
                    "confidence_reason": {"type": "string"},
                    "risk_hint": {"type": "string"}
                },
                "required": ["clinical_reason", "confidence_reason", "risk_hint"]
            }
            data = await self._call_solar_json(
                prompt=prompt,
                schema_name="why_card",
                json_schema=schema,
                max_tokens=3000,
                retries=2
            )
            meta = data.get("_meta", {})
            return {
                "clinical_reason": data.get("clinical_reason", ""),
                "confidence_reason": data.get("confidence_reason", ""),
                "risk_hint": data.get("risk_hint", ""),
                "generated_by_llm": bool(meta.get("generated_by_llm", True)),
                "generator": meta.get("generator", "Solar Pro 3"),
                "status": "ok"
            }
        except Exception as e:
            logger.warning(f"explain_nodule_why fallback: {e}")
            if llm_only:
                return {
                    "clinical_reason": "LLM generation failed",
                    "confidence_reason": "LLM generation failed",
                    "risk_hint": "LLM generation failed",
                    "generated_by_llm": False,
                    "generator": "Solar Pro 3",
                    "status": "failed"
                }
            conf = float(nodule.get("confidence", 0.0))
            size = float(nodule.get("diameter_mm", 0.0))
            vol = float(nodule.get("volume_mm3", 0.0))
            loc = str(nodule.get("location_code", "UNK"))
            srange = nodule.get("slice_range", [0, 0])
            return {
                "clinical_reason": f"{loc} 위치에서 직경 {size:.1f}mm, 부피 {vol:.1f}mm³ 후보가 slice {srange[0]}-{srange[1]} 범위에 연속 검출되었습니다.",
                "confidence_reason": f"confidence {conf:.2f}는 현재 운영 임계값(약 0.15) 대비 {'높은' if conf >= 0.25 else '경계'} 수준입니다.",
                "risk_hint": "해당 slice 범위에서 혈관 겹침/흉막 인접 여부와 형태 연속성을 우선 확인하세요.",
                "generated_by_llm": False,
                "generator": "Rule Fallback",
                "status": "fallback"
            }

    async def narrate_prior_delta(self, current: Dict, prior: Dict, llm_only: bool = False) -> Dict:
        """Narrate prior-vs-current change with quantitative context."""
        if self.use_mock:
            return {
                "summary": current.get("summary_seed", "현재/과거 비교 요약 정보가 부족합니다."),
                "change_type": current.get("change_type", "UNKNOWN"),
                "clinical_note": "수치 변화와 slice 위치 일치 여부를 함께 확인하세요.",
                "generated_by_llm": False,
                "generator": "MOCK",
                "status": "mock"
            }

        prompt = self.PRIOR_DELTA_PROMPT.format(current=str(current), prior=str(prior))
        try:
            schema = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {"type": "string"},
                    "change_type": {"type": "string"},
                    "clinical_note": {"type": "string"}
                },
                "required": ["summary", "change_type", "clinical_note"]
            }
            data = await self._call_solar_json(
                prompt=prompt,
                schema_name="prior_delta",
                json_schema=schema,
                max_tokens=3000,
                retries=2
            )
            meta = data.get("_meta", {})
            return {
                "summary": data.get("summary", ""),
                "change_type": data.get("change_type", "UNKNOWN"),
                "clinical_note": data.get("clinical_note", ""),
                "generated_by_llm": bool(meta.get("generated_by_llm", True)),
                "generator": meta.get("generator", "Solar Pro 3"),
                "status": "ok"
            }
        except Exception as e:
            logger.warning(f"narrate_prior_delta fallback: {e}")
            if llm_only:
                return {
                    "summary": "LLM generation failed",
                    "change_type": "UNKNOWN",
                    "clinical_note": "LLM generation failed",
                    "generated_by_llm": False,
                    "generator": "Solar Pro 3",
                    "status": "failed"
                }
            c_size = current.get("largest_nodule_mm")
            p_size = prior.get("prior_largest_mm")
            if isinstance(c_size, (int, float)) and isinstance(p_size, (int, float)) and p_size > 0:
                delta_mm = c_size - p_size
                delta_pct = (delta_mm / p_size) * 100.0
                change = "GROWTH" if delta_pct > 20 else "REDUCTION" if delta_pct < -20 else "STABLE"
                summary = (
                    f"현재 최대 결절은 {c_size:.1f}mm, 과거는 {p_size:.1f}mm로 "
                    f"{delta_mm:+.1f}mm({delta_pct:+.1f}%) 변화입니다."
                )
                note = "증가 폭이 크면 동일 병변 여부를 slice 위치와 함께 재확인하세요." if change == "GROWTH" else "수치 변화가 경미한지 prior 위치 매칭으로 확인하세요."
            else:
                change = current.get("change_type", "UNKNOWN")
                summary = "과거 비교 수치가 제한적이라 정량 변화 판정이 어렵습니다."
                note = "prior 메타데이터(날짜/직경) 유무를 먼저 확인하세요."
            return {
                "summary": summary,
                "change_type": change,
                "clinical_note": note,
                "generated_by_llm": False,
                "generator": "Rule Fallback",
                "status": "fallback"
            }

    async def suggest_action(self, evidence: Dict, llm_only: bool = False) -> Dict:
        """Generate action suggestion with explicit numeric rationale."""
        if self.use_mock:
            sens = evidence.get("sensitivity_hint")
            return {
                "recommendation_level": "review_now" if evidence.get("max_confidence", 0) >= 0.25 else "monitor",
                "rationale": f"최대 결절 confidence={evidence.get('max_confidence', 0):.2f}, 직경={evidence.get('max_diameter_mm', 0):.1f}mm를 근거로 판독 우선순위를 제안합니다."
                + (f" 현재 설정 sensitivity는 약 {sens:.2f} 수준입니다." if isinstance(sens, (int, float)) else ""),
                "next_step": "오버레이에서 상위 후보를 먼저 확인하고 prior 비교 결과를 함께 검토하세요.",
                "generated_by_llm": False,
                "generator": "MOCK",
                "status": "mock"
            }

        prompt = self.ACTION_SUGGEST_PROMPT.format(evidence=str(evidence))
        try:
            schema = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "recommendation_level": {"type": "string"},
                    "rationale": {"type": "string"},
                    "next_step": {"type": "string"}
                },
                "required": ["recommendation_level", "rationale", "next_step"]
            }
            data = await self._call_solar_json(
                prompt=prompt,
                schema_name="action_suggestion",
                json_schema=schema,
                max_tokens=3000,
                retries=2
            )
            meta = data.get("_meta", {})
            return {
                "recommendation_level": data.get("recommendation_level", "monitor"),
                "rationale": data.get("rationale", ""),
                "next_step": data.get("next_step", ""),
                "generated_by_llm": bool(meta.get("generated_by_llm", True)),
                "generator": meta.get("generator", "Solar Pro 3"),
                "status": "ok"
            }
        except Exception as e:
            logger.warning(f"suggest_action fallback: {e}")
            if llm_only:
                return {
                    "recommendation_level": "unavailable",
                    "rationale": "LLM generation failed",
                    "next_step": "LLM generation failed",
                    "generated_by_llm": False,
                    "generator": "Solar Pro 3",
                    "status": "failed"
                }
            max_conf = float(evidence.get("max_confidence", 0.0))
            max_d = float(evidence.get("max_diameter_mm", 0.0))
            n_count = int(evidence.get("nodule_count", 0))
            sens = evidence.get("sensitivity_hint")
            prior_delta = evidence.get("prior_delta", {}) if isinstance(evidence.get("prior_delta"), dict) else {}
            change_type = str(prior_delta.get("change_type", "UNKNOWN"))
            delta_pct = prior_delta.get("delta_pct")

            if change_type == "GROWTH" and (isinstance(delta_pct, (int, float)) and delta_pct > 20):
                level = "review_now"
            elif max_d >= 10.0 or max_conf >= 0.30:
                level = "routine_followup"
            else:
                level = "monitor"

            sens_text = f", sensitivity≈{float(sens):.3f}" if isinstance(sens, (int, float)) else ""
            delta_text = (
                f", prior change={float(delta_pct):+.1f}%({change_type})"
                if isinstance(delta_pct, (int, float))
                else ""
            )
            rationale = (
                f"후보 {n_count}개 중 최대 결절 직경={max_d:.1f}mm, 최대 confidence={max_conf:.2f}"
                f"{delta_text}{sens_text}를 기준으로 우선순위를 분류했습니다."
            )
            if level == "review_now":
                next_step = "상위 후보를 오버레이에서 즉시 재확인하고 prior 동일 병변 매칭을 우선 검토하세요."
            elif level == "routine_followup":
                next_step = "보고서 작성 전 상위 후보의 형태/연속성을 확인한 뒤 추적 비교를 함께 기록하세요."
            else:
                next_step = "현재는 모니터링 우선으로 두고 신규/증가 패턴이 있는지 주기적으로 재평가하세요."
            return {
                "recommendation_level": level,
                "rationale": rationale,
                "next_step": next_step,
                "generated_by_llm": False,
                "generator": "Rule Fallback",
                "status": "fallback"
            }

    async def explain_threshold_tradeoff(
        self,
        selected: Dict,
        best_screening: Dict,
        best_reporting: Dict,
        mode: str,
        llm_only: bool = False
    ) -> Dict:
        """Explain why selected threshold fits screening/reporting context."""
        if self.use_mock:
            s_thr = float(selected.get("threshold", 0.0))
            s_sens = float(selected.get("sensitivity", 0.0))
            s_fp = float(selected.get("fp_per_scan", 0.0))
            return {
                "text": (
                    f"선택 임계값(th={s_thr:.2f})에서 sensitivity={s_sens:.3f}, "
                    f"FP/scan={s_fp:.2f}입니다. "
                    f"{'Screening' if mode == 'screening' else 'Reporting' if mode == 'reporting' else 'Balanced'} 목적에서는 "
                    f"민감도와 오탐의 균형을 이 수치로 판단할 수 있습니다."
                ),
                "generated_by_llm": False,
                "generator": "MOCK",
                "status": "mock"
            }

        prompt = self.THRESHOLD_EXPLAIN_PROMPT.format(
            selected=str(selected),
            best_screening=str(best_screening),
            best_reporting=str(best_reporting),
            mode=mode
        )
        try:
            schema = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
            data = await self._call_solar_json(
                prompt=prompt,
                schema_name="threshold_tradeoff",
                json_schema=schema,
                max_tokens=3000,
                retries=2
            )
            meta = data.get("_meta", {})
            text = str(data.get("text", "")).strip()
            if not text:
                raise ValueError("Empty response content")
            return {
                "text": text,
                "generated_by_llm": bool(meta.get("generated_by_llm", True)),
                "generator": meta.get("generator", "Solar Pro 3"),
                "status": "ok"
            }
        except Exception as e:
            logger.warning(f"explain_threshold_tradeoff fallback: {e}")
            if llm_only:
                return {
                    "text": "LLM generation failed",
                    "generated_by_llm": False,
                    "generator": "Solar Pro 3",
                    "status": "failed"
                }
            s_thr = float(selected.get("threshold", 0.0))
            s_sens = float(selected.get("sensitivity", 0.0))
            s_fp = float(selected.get("fp_per_scan", 0.0))
            sc_sens = float(best_screening.get("sensitivity", 0.0))
            sc_fp = float(best_screening.get("fp_per_scan", 0.0))
            rp_sens = float(best_reporting.get("sensitivity", 0.0))
            rp_fp = float(best_reporting.get("fp_per_scan", 0.0))
            return {
                "text": (
                    f"선택값(th={s_thr:.2f})은 sensitivity={s_sens:.3f}, FP/scan={s_fp:.2f}입니다. "
                    f"Screening 기준 최적은 sensitivity={sc_sens:.3f}, FP/scan={sc_fp:.2f}이고, "
                    f"Reporting 기준 최적은 sensitivity={rp_sens:.3f}, FP/scan={rp_fp:.2f}입니다. "
                    f"따라서 현재 설정은 {('민감도 우선' if mode == 'screening' else '오탐 억제 우선' if mode == 'reporting' else '균형형')} 운영에서 중간 절충점으로 해석할 수 있습니다."
                ),
                "generated_by_llm": False,
                "generator": "Rule Fallback",
                "status": "fallback"
            }
    
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
