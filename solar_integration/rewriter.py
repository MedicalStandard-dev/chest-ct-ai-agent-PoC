# solar_integration/rewriter.py
"""
Solar Pro 3 Narrative-Only Rewriter (v3.0)

핵심 원칙:
- Solar는 표/수치를 절대 보지 못함
- 비표(narrative) 텍스트만 추출하여 전달
- "Solar가 뭘 했는지"가 명확함

방식 (Narrative-Only, 기본):
1. 비표 텍스트만 추출 (QUALITY, LIMITATIONS, NOTES, PRIOR 요약)
2. 이것만 Solar에게 전달
3. 결과를 원본 리포트에 삽입
4. 표/수치는 전혀 건드리지 않음

방식 (Placeholder, 레거시):
1. 표 블록을 placeholder로 치환
2. Solar에게 전달
3. 복원 및 검증
"""
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
import re
import hashlib
import httpx
from config.settings import settings
from utils.logger import logger


@dataclass
class NarrativeSegment:
    """비표 텍스트 세그먼트"""
    key: str                    # 식별자 (quality, limitations, notes, prior_summary)
    original_text: str          # 원본 텍스트
    rewritten_text: str = ""    # 재작성된 텍스트
    start_line: int = 0         # 원본에서의 시작 라인
    end_line: int = 0           # 원본에서의 종료 라인


class NarrativeExtractor:
    """
    비표(narrative) 텍스트만 추출하는 클래스
    
    추출 대상:
    - QUALITY / LIMITATIONS 섹션
    - NOTES 섹션
    - PRIOR 요약 문장 (표 외부)
    
    제외 대상:
    - FINDINGS TABLE
    - MEASUREMENTS TABLE
    - PRIOR COMPARISON TABLE
    - KEY FLAGS
    - AUDIT
    """
    
    # 비표 텍스트 섹션 (Solar가 다듬을 수 있음)
    NARRATIVE_SECTIONS = {
        "quality": {
            "headers": ["QUALITY / LIMITATIONS", "QUALITY/LIMITATIONS", "QUALITY"],
            "description": "스캔 품질 및 제한사항"
        },
        "notes": {
            "headers": ["NOTES", "NOTE"],
            "description": "경고문 및 안내 문구"
        },
        "header": {
            "headers": ["DRAFT", "CT Chest"],
            "description": "리포트 헤더 (날짜/환자 정보 제외)"
        }
    }
    
    # 표 섹션 (Solar가 건드리면 안 됨)
    TABLE_SECTIONS = [
        "FINDINGS - TABLE", "FINDINGS TABLE",
        "MEASUREMENTS - TABLE", "MEASUREMENTS TABLE", 
        "PRIOR COMPARISON - TABLE", "PRIOR COMPARISON TABLE",
        "KEY FLAGS", "AUDIT"
    ]
    
    def __init__(self):
        self.segments: Dict[str, NarrativeSegment] = {}
        self.original_lines: List[str] = []
    
    def extract_narratives(self, report_text: str) -> Dict[str, NarrativeSegment]:
        """
        리포트에서 비표 텍스트만 추출
        
        Returns:
            Dict[key, NarrativeSegment]
        """
        self.segments = {}
        self.original_lines = report_text.splitlines()
        
        for section_key, section_info in self.NARRATIVE_SECTIONS.items():
            segment = self._extract_section(section_info["headers"])
            if segment:
                segment.key = section_key
                self.segments[section_key] = segment
        
        logger.info(f"Extracted {len(self.segments)} narrative segments")
        return self.segments
    
    def _extract_section(self, headers: List[str]) -> Optional[NarrativeSegment]:
        """섹션 추출"""
        start_idx = None
        
        for i, line in enumerate(self.original_lines):
            for header in headers:
                if header in line:
                    start_idx = i
                    break
            if start_idx is not None:
                break
        
        if start_idx is None:
            return None
        
        # 다음 주요 섹션 또는 표 섹션까지
        end_idx = start_idx + 1
        while end_idx < len(self.original_lines):
            line = self.original_lines[end_idx]
            
            # 표 섹션 또는 다른 주요 섹션 만나면 종료
            if any(ts in line for ts in self.TABLE_SECTIONS):
                break
            if "━━━" in line and end_idx > start_idx + 1:
                break
            
            end_idx += 1
        
        content = "\n".join(self.original_lines[start_idx:end_idx]).strip()
        
        if not content:
            return None
        
        return NarrativeSegment(
            key="",
            original_text=content,
            start_line=start_idx,
            end_line=end_idx
        )
    
    def apply_rewrites(self, report_text: str, rewritten_segments: Dict[str, str]) -> str:
        """
        재작성된 비표 텍스트를 원본 리포트에 적용
        
        Args:
            report_text: 원본 리포트
            rewritten_segments: {key: rewritten_text}
            
        Returns:
            재작성된 텍스트가 적용된 리포트
        """
        result_lines = report_text.splitlines()
        
        # 역순으로 적용 (라인 번호 밀림 방지)
        sorted_segments = sorted(
            self.segments.items(), 
            key=lambda x: x[1].start_line, 
            reverse=True
        )
        
        for key, segment in sorted_segments:
            if key in rewritten_segments:
                new_text = rewritten_segments[key]
                new_lines = new_text.splitlines()
                
                # 해당 범위 교체
                result_lines[segment.start_line:segment.end_line] = new_lines
        
        return "\n".join(result_lines)
    
    def get_combined_narrative_text(self) -> str:
        """
        추출된 모든 비표 텍스트를 하나로 결합
        Solar에게 전달할 텍스트
        """
        parts = []
        for key, segment in self.segments.items():
            parts.append(f"[{key.upper()}]\n{segment.original_text}")
        
        return "\n\n---\n\n".join(parts)


class PlaceholderManager:
    """표 블록을 placeholder로 치환/복원하는 관리자"""
    
    # Placeholder 토큰 정의
    PLACEHOLDERS = {
        "findings": "[[TABLE_FINDINGS]]",
        "measurements": "[[TABLE_MEASUREMENTS]]",
        "prior_comparison": "[[TABLE_PRIOR]]",
        "key_flags": "[[KEY_FLAGS]]",
        "quality": "[[QUALITY_LIMITATIONS]]",
        "audit": "[[AUDIT_INFO]]"
    }
    
    # 섹션 헤더 매핑 (리포트에서 탐지용)
    SECTION_HEADERS = {
        "findings": ["FINDINGS - TABLE", "FINDINGS TABLE", "📊 FINDINGS"],
        "measurements": ["MEASUREMENTS - TABLE", "MEASUREMENTS TABLE", "📏 MEASUREMENTS"],
        "prior_comparison": ["PRIOR COMPARISON - TABLE", "PRIOR COMPARISON TABLE", "🔄 PRIOR COMPARISON"],
        "key_flags": ["KEY FLAGS"],
        "quality": ["QUALITY / LIMITATIONS", "QUALITY/LIMITATIONS"],
        "audit": ["AUDIT"]
    }
    
    def __init__(self):
        self.extracted_blocks: Dict[str, str] = {}
        self.block_hashes: Dict[str, str] = {}
    
    def extract_and_replace(self, report_text: str) -> Tuple[str, Dict[str, str]]:
        """
        리포트에서 표 블록을 추출하고 placeholder로 치환
        
        Args:
            report_text: 원본 리포트 텍스트
            
        Returns:
            (placeholder가 적용된 텍스트, 추출된 블록 딕셔너리)
        """
        self.extracted_blocks = {}
        self.block_hashes = {}
        
        result_text = report_text
        
        for block_type, headers in self.SECTION_HEADERS.items():
            block_content = self._extract_block(result_text, headers)
            if block_content:
                self.extracted_blocks[block_type] = block_content
                self.block_hashes[block_type] = self._compute_hash(block_content)
                
                # Placeholder로 치환
                placeholder = self.PLACEHOLDERS[block_type]
                result_text = self._replace_block(result_text, headers, placeholder)
        
        logger.info(f"Extracted {len(self.extracted_blocks)} blocks for placeholder protection")
        return result_text, self.extracted_blocks
    
    def restore_placeholders(self, rewritten_text: str) -> str:
        """
        Placeholder를 원래 표 블록으로 복원
        
        Args:
            rewritten_text: Solar가 rewrite한 텍스트
            
        Returns:
            표가 복원된 텍스트
        """
        result = rewritten_text
        
        for block_type, placeholder in self.PLACEHOLDERS.items():
            if block_type in self.extracted_blocks:
                original_block = self.extracted_blocks[block_type]
                # Placeholder를 원래 블록으로 교체
                result = result.replace(placeholder, original_block)
        
        return result
    
    def verify_placeholder_integrity(self, rewritten_text: str) -> Tuple[bool, List[str]]:
        """
        Placeholder가 손상되지 않았는지 검증
        
        Returns:
            (통과 여부, 에러 메시지 리스트)
        """
        errors = []
        
        for block_type, placeholder in self.PLACEHOLDERS.items():
            if block_type in self.extracted_blocks:
                # Placeholder가 존재해야 함
                if placeholder not in rewritten_text:
                    errors.append(f"Placeholder missing: {placeholder}")
                    continue
                
                # Placeholder가 변형되지 않았는지 확인 (공백 추가 등)
                # 유연한 매칭: [[ TABLE_FINDINGS ]] 같은 변형 탐지
                pattern = re.escape(placeholder).replace(r'\[\[', r'\[\s*\[').replace(r'\]\]', r'\]\s*\]')
                matches = re.findall(pattern, rewritten_text)
                
                if len(matches) != 1:
                    if len(matches) == 0:
                        errors.append(f"Placeholder corrupted or missing: {placeholder}")
                    else:
                        errors.append(f"Placeholder duplicated: {placeholder} (found {len(matches)} times)")
        
        return len(errors) == 0, errors
    
    def verify_restored_hashes(self, restored_text: str) -> Tuple[bool, List[str]]:
        """
        복원 후 표 블록의 해시가 동일한지 검증
        
        Returns:
            (통과 여부, 에러 메시지 리스트)
        """
        errors = []
        
        for block_type, original_hash in self.block_hashes.items():
            # 복원된 텍스트에서 해당 블록 추출
            headers = self.SECTION_HEADERS[block_type]
            restored_block = self._extract_block(restored_text, headers)
            
            if not restored_block:
                errors.append(f"Block not found after restoration: {block_type}")
                continue
            
            restored_hash = self._compute_hash(restored_block)
            if restored_hash != original_hash:
                errors.append(f"Block hash mismatch: {block_type}")
        
        return len(errors) == 0, errors
    
    def _extract_block(self, text: str, headers: List[str]) -> Optional[str]:
        """섹션 헤더를 기준으로 블록 추출"""
        lines = text.splitlines()
        
        start_idx = None
        for i, line in enumerate(lines):
            for header in headers:
                if header in line:
                    start_idx = i
                    break
            if start_idx is not None:
                break
        
        if start_idx is None:
            return None
        
        # 다음 섹션 또는 빈 줄까지 추출
        end_idx = start_idx + 1
        blank_count = 0
        
        while end_idx < len(lines):
            line = lines[end_idx]
            
            # 다른 주요 섹션 헤더 만나면 종료
            if any(h in line for h in ["FINDINGS", "MEASUREMENTS", "PRIOR", "KEY FLAGS", 
                                        "QUALITY", "AUDIT", "NOTES", "━━━"]):
                if end_idx > start_idx + 1:  # 최소 1줄은 있어야
                    break
            
            # 연속 빈 줄 2개 이상이면 종료
            if line.strip() == "":
                blank_count += 1
                if blank_count >= 2:
                    break
            else:
                blank_count = 0
            
            end_idx += 1
        
        block_lines = lines[start_idx:end_idx]
        return "\n".join(block_lines).strip()
    
    def _replace_block(self, text: str, headers: List[str], placeholder: str) -> str:
        """블록을 placeholder로 교체"""
        lines = text.splitlines()
        
        start_idx = None
        for i, line in enumerate(lines):
            for header in headers:
                if header in line:
                    start_idx = i
                    break
            if start_idx is not None:
                break
        
        if start_idx is None:
            return text
        
        # 다음 섹션까지 범위 찾기
        end_idx = start_idx + 1
        blank_count = 0
        
        while end_idx < len(lines):
            line = lines[end_idx]
            
            if any(h in line for h in ["FINDINGS", "MEASUREMENTS", "PRIOR", "KEY FLAGS", 
                                        "QUALITY", "AUDIT", "NOTES", "━━━"]):
                if end_idx > start_idx + 1:
                    break
            
            if line.strip() == "":
                blank_count += 1
                if blank_count >= 2:
                    break
            else:
                blank_count = 0
            
            end_idx += 1
        
        # 블록을 placeholder로 교체
        new_lines = lines[:start_idx] + [placeholder] + lines[end_idx:]
        return "\n".join(new_lines)
    
    def _compute_hash(self, content: str) -> str:
        """블록 내용의 SHA256 해시 계산"""
        # 공백 정규화 후 해시
        normalized = " ".join(content.split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]


class SolarProRewriter:
    """
    Solar Pro 3 Narrative-Only Rewriter (v3.0)
    
    핵심 원칙:
    - Solar는 표/수치를 절대 보지 못함
    - 비표 텍스트만 전달하여 "협업감" 극대화
    - 검증 실패 시 원본 반환 (fail-closed)
    
    Solar가 하는 일:
    - QUALITY/LIMITATIONS 문장 다듬기
    - NOTES 문구 표준화
    - 중복 제거, 톤 통일
    
    Solar가 못 하는 일:
    - 표 보기/수정 (아예 안 줌)
    - 숫자/단위 변경 (추출 안 함)
    - 소견 추가/삭제
    """
    
    # Narrative-Only 프롬프트 (기본)
    NARRATIVE_SYSTEM_PROMPT = """You are a PACS report text polisher.
You will receive ONLY the narrative text sections (not tables or measurements).
Your job: make these sentences short, clear, and clinical.

Rules:
- Remove redundant sentences
- Standardize warning phrases
- Keep the same meaning
- DO NOT add clinical findings
- DO NOT infer diagnosis
- Return the text in the same [SECTION] format"""

    NARRATIVE_USER_PROMPT_TEMPLATE = """Polish these narrative sections for a PACS report.
Keep them short and operational. Remove redundancy.

Input:
<<<
{narrative_text}
>>>

Return in the same format:
[QUALITY]
...
[NOTES]
..."""

    # Placeholder 프롬프트 (레거시)
    PLACEHOLDER_SYSTEM_PROMPT = """You are a PACS report rewriting module (rewrite-only).
You must preserve all placeholders exactly as-is.
Do not add, remove, or change any clinical content.
Do not infer diagnosis or recommend treatment.
Return the full report text with the same structure."""

    PLACEHOLDER_USER_PROMPT_TEMPLATE = """Rewrite only the non-table narrative text to be short and operational (PACS style).
Rules:
- DO NOT modify any placeholders like [[TABLE_*]], [[KEY_FLAGS]], [[QUALITY_LIMITATIONS]], [[AUDIT_INFO]].
- DO NOT change any numbers, units, dates, laterality, or locations.
- DO NOT add new findings or delete existing findings.
- If a sentence cannot be supported by the provided evidence text, remove it.
- Keep tone concise and clinical.

Report:
<<<
{report_with_placeholders}
>>>"""

    SUMMARY_SYSTEM_PROMPT = """You are an AI summarization module for medical PACS.
You MUST NOT:
- Add new findings
- Infer diagnosis
- Use clinical judgment
- Recommend actions

You MAY:
- Rephrase and summarize existing table facts
- Use only provided structured data

Output format:
SUMMARY
- bullet 1
- bullet 2
- bullet 3 (optional)
"""

    SUMMARY_USER_PROMPT_TEMPLATE = """Generate a short factual summary using ONLY the following facts:

KEY FLAGS
{key_flags}

FINDINGS TABLE (text facts)
{findings_table_text}

PRIOR COMPARISON TABLE (text facts)
{prior_table_text}
"""
    
    def __init__(self, api_key: Optional[str] = None, use_mock: bool = False):
        """
        Args:
            api_key: Solar API key
            use_mock: True면 mock rewrite (no API call)
        """
        self.api_key = api_key or getattr(settings, 'solar_api_key', None)
        self.endpoint = getattr(settings, 'solar_api_endpoint', 'https://api.upstage.ai/v1/solar')
        self.model_name = getattr(settings, 'solar_report_model', 'solar-pro')
        self.use_mock = use_mock or (self.api_key is None)
        
        self.placeholder_manager = PlaceholderManager()
        self.narrative_extractor = NarrativeExtractor()
        
        if self.use_mock:
            logger.warning("Solar Pro Rewriter: Using MOCK mode (passthrough)")
        else:
            logger.info("Solar Pro Rewriter: Using REAL API (narrative-only mode)")
    
    # ========== 기본 방식: Narrative-Only ==========
    
    async def rewrite_narrative_only(
        self,
        report_text: str
    ) -> Tuple[str, Dict]:
        """
        비표 텍스트만 추출하여 Solar에게 전달 (권장 방식)
        
        Solar는 QUALITY, NOTES 등 narrative 텍스트만 봄.
        표/수치/소견은 전혀 전달되지 않음.
        
        Args:
            report_text: 원본 리포트 (표 포함)
            
        Returns:
            (rewritten_report, audit_info)
        """
        audit = {
            "mode": "narrative-only",
            "segments_extracted": 0,
            "segments_rewritten": 0,
            "validation_passed": False,
            "fallback_used": False,
            "solar_input_chars": 0,
            "errors": []
        }
        
        # 1. 비표 텍스트만 추출
        segments = self.narrative_extractor.extract_narratives(report_text)
        audit["segments_extracted"] = len(segments)
        
        if not segments:
            logger.info("No narrative segments found, returning original")
            audit["validation_passed"] = True
            return report_text, audit
        
        if self.use_mock:
            # Mock: 원본 그대로 반환
            audit["validation_passed"] = True
            return report_text, audit
        
        try:
            # 2. 비표 텍스트만 Solar에게 전달
            narrative_text = self.narrative_extractor.get_combined_narrative_text()
            audit["solar_input_chars"] = len(narrative_text)
            
            logger.info(f"Sending {len(narrative_text)} chars of narrative text to Solar")
            
            rewritten_narrative = await self._call_solar_narrative_api(narrative_text)
            
            # 3. 응답 파싱
            rewritten_segments = self._parse_narrative_response(rewritten_narrative)
            audit["segments_rewritten"] = len(rewritten_segments)
            
            # 4. 검증: 숫자/날짜가 추가되지 않았는지
            validation_ok, validation_errors = self._validate_narrative_response(
                segments, rewritten_segments
            )
            
            if not validation_ok:
                audit["errors"].extend(validation_errors)
                audit["fallback_used"] = True
                logger.warning(f"Narrative validation failed: {validation_errors}")
                return report_text, audit  # Fail-closed
            
            # 5. 원본 리포트에 적용
            final_report = self.narrative_extractor.apply_rewrites(
                report_text, rewritten_segments
            )
            
            audit["validation_passed"] = True
            logger.info("Narrative-only rewrite completed successfully")
            return final_report, audit
            
        except Exception as e:
            logger.error(f"Narrative rewrite failed: {e}")
            audit["errors"].append(str(e))
            audit["fallback_used"] = True
            return report_text, audit  # Fail-closed
    
    async def _call_solar_narrative_api(self, narrative_text: str) -> str:
        """Solar API 호출 (narrative-only)"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        user_prompt = self.NARRATIVE_USER_PROMPT_TEMPLATE.format(
            narrative_text=narrative_text
        )
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.NARRATIVE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000,  # narrative만이라 적음
            "top_p": 0.9
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.endpoint}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            rewritten = result["choices"][0]["message"]["content"].strip()
            
            logger.info("Solar narrative API call completed")
            return rewritten
    
    def _parse_narrative_response(self, response: str) -> Dict[str, str]:
        """Solar 응답에서 섹션별 텍스트 파싱"""
        segments = {}
        
        # [QUALITY], [NOTES] 등의 패턴으로 파싱
        pattern = r'\[([A-Z_]+)\]\s*\n(.*?)(?=\n\[[A-Z_]+\]|\Z)'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for key, content in matches:
            segments[key.lower()] = content.strip()
        
        return segments
    
    def _validate_narrative_response(
        self,
        original_segments: Dict[str, NarrativeSegment],
        rewritten_segments: Dict[str, str]
    ) -> Tuple[bool, List[str]]:
        """
        narrative 응답 검증
        - 새로운 숫자/단위가 추가되지 않았는지
        - 금지 표현이 없는지
        """
        errors = []
        
        # 숫자 패턴
        number_pattern = r'\d+\.?\d*\s*(mm|cm|mm³|%)'
        
        for key, rewritten in rewritten_segments.items():
            if key not in original_segments:
                continue
            
            original = original_segments[key].original_text
            
            # 원본에 없던 숫자/단위가 추가되었는지
            original_numbers = set(re.findall(number_pattern, original, re.IGNORECASE))
            rewritten_numbers = set(re.findall(number_pattern, rewritten, re.IGNORECASE))
            
            new_numbers = rewritten_numbers - original_numbers
            if new_numbers:
                errors.append(f"New measurements added in {key}: {new_numbers}")
            
            # 금지 표현 체크
            forbidden = ["diagnosis", "diagnosed", "rule out", "recommend", "treatment"]
            for term in forbidden:
                if term.lower() in rewritten.lower() and term.lower() not in original.lower():
                    errors.append(f"Forbidden term added in {key}: {term}")
        
        return len(errors) == 0, errors
    
    # ========== 레거시 방식: Placeholder ==========
    
    async def rewrite_report_protected(
        self, 
        report_text: str
    ) -> Tuple[str, Dict]:
        """
        Table-protected 방식으로 리포트 rewrite
        
        Args:
            report_text: 원본 리포트 (표 포함)
            
        Returns:
            (rewritten_report, audit_info)
        """
        audit = {
            "mode": "table-protected-rewrite",
            "placeholder_count": 0,
            "validation_passed": False,
            "fallback_used": False,
            "errors": []
        }
        
        # 1. 표 블록 추출 및 placeholder 치환
        text_with_placeholders, extracted = self.placeholder_manager.extract_and_replace(report_text)
        audit["placeholder_count"] = len(extracted)
        
        if self.use_mock:
            # Mock: 원본 그대로 반환
            audit["validation_passed"] = True
            return report_text, audit
        
        try:
            # 2. Solar API 호출 (placeholder 포함 텍스트만)
            rewritten_with_placeholders = await self._call_solar_placeholder_api(text_with_placeholders)
            
            # 3. Placeholder 무결성 검증
            integrity_ok, integrity_errors = self.placeholder_manager.verify_placeholder_integrity(
                rewritten_with_placeholders
            )
            
            if not integrity_ok:
                audit["errors"].extend(integrity_errors)
                audit["fallback_used"] = True
                logger.warning(f"Placeholder integrity failed: {integrity_errors}")
                return report_text, audit  # Fail-closed: 원본 반환
            
            # 4. Placeholder 복원
            restored_report = self.placeholder_manager.restore_placeholders(rewritten_with_placeholders)
            
            # 5. 복원 후 해시 검증
            hash_ok, hash_errors = self.placeholder_manager.verify_restored_hashes(restored_report)
            
            if not hash_ok:
                audit["errors"].extend(hash_errors)
                audit["fallback_used"] = True
                logger.warning(f"Hash verification failed: {hash_errors}")
                return report_text, audit  # Fail-closed: 원본 반환
            
            audit["validation_passed"] = True
            logger.info("Table-protected rewrite completed successfully")
            return restored_report, audit
            
        except Exception as e:
            logger.error(f"Solar Pro rewrite failed: {e}")
            audit["errors"].append(str(e))
            audit["fallback_used"] = True
            return report_text, audit  # Fail-closed: 원본 반환
    
    async def _call_solar_placeholder_api(self, text_with_placeholders: str) -> str:
        """Solar API 호출 (placeholder 방식)"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        user_prompt = self.PLACEHOLDER_USER_PROMPT_TEMPLATE.format(
            report_with_placeholders=text_with_placeholders
        )
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.PLACEHOLDER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 3000,
            "top_p": 0.9
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.endpoint}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            rewritten = result["choices"][0]["message"]["content"].strip()
            
            logger.info("Solar placeholder API call completed")
            return rewritten

    async def generate_ai_summary(
        self,
        key_flags_text: str,
        findings_table_text: str,
        prior_table_text: str
    ) -> str:
        """KEY FLAGS + 테이블 기반 요약 생성"""
        if self.use_mock:
            return ""
        
        try:
            user_prompt = self.SUMMARY_USER_PROMPT_TEMPLATE.format(
                key_flags=key_flags_text,
                findings_table_text=findings_table_text,
                prior_table_text=prior_table_text
            )
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1000,
                "top_p": 0.9
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.endpoint}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                
                if not result.get("choices"):
                    logger.warning("AI Summary: Solar returned no choices")
                    return ""
                
                summary = result["choices"][0].get("message", {}).get("content", "")
                if not summary:
                    logger.warning("AI Summary: Solar returned empty content")
                
                return summary.strip()
        except Exception as e:
            logger.error("AI Summary generation failed: {}", str(e))
            return ""
    
    # ========== 통합 인터페이스 ==========
    
    async def rewrite_report_text(
        self, 
        report_text: str, 
        mode: str = "narrative"
    ) -> str:
        """
        리포트 텍스트 rewrite (통합 인터페이스)
        
        Args:
            report_text: 원본 리포트
            mode: "narrative" (기본, 권장) 또는 "placeholder" (레거시)
            
        Returns:
            재작성된 리포트
        """
        if mode == "narrative":
            result, _ = await self.rewrite_narrative_only(report_text)
        else:
            result, _ = await self.rewrite_report_protected(report_text)
        return result
    
    async def rewrite_with_audit(
        self,
        report_text: str,
        mode: str = "narrative"
    ) -> Tuple[str, Dict]:
        """
        리포트 텍스트 rewrite (감사 정보 포함)
        
        Args:
            report_text: 원본 리포트
            mode: "narrative" (기본) 또는 "placeholder"
            
        Returns:
            (재작성된 리포트, 감사 정보)
        """
        if mode == "narrative":
            return await self.rewrite_narrative_only(report_text)
        else:
            return await self.rewrite_report_protected(report_text)
    
    # Legacy (backward compatibility)
    async def rewrite(self, template_text: str, section_name: str = "") -> str:
        """Legacy: 단순 rewrite"""
        if self.use_mock:
            return template_text
        
        try:
            return await self._call_solar_narrative_api(template_text)
        except Exception as e:
            logger.error(f"Solar Pro rewrite failed: {e}")
            return template_text
    
    def get_rewriter_info(self) -> dict:
        """Rewriter 정보 반환"""
        return {
            "provider": "Solar Pro 3" if not self.use_mock else "Mock",
            "mode": "narrative-only",  # 기본 모드
            "alternative_mode": "placeholder",
            "mock": self.use_mock,
            "version": "v3.0",
            "description": "Solar는 QUALITY, NOTES 등 비표 텍스트만 다듬음. 표/수치는 전혀 전달 안 됨."
        }
