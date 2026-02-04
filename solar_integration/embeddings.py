# solar_integration/embeddings.py
"""
Solar Embedding API client for RAG retrieval
SOLAR_API_KEY 없으면 deterministic mock 사용
"""
from typing import List, Optional
import hashlib
import numpy as np
import httpx
from config.settings import settings
from utils.logger import logger


class SolarEmbeddingClient:
    """Solar Embedding API wrapper"""
    
    def __init__(self, api_key: Optional[str] = None, use_mock: bool = False):
        """
        Args:
            api_key: Solar API key
            use_mock: True면 mock embedding 강제 사용
        """
        self.api_key = api_key or getattr(settings, "solar_api_key", None)
        self.endpoint = getattr(settings, "solar_api_endpoint", "https://api.upstage.ai/v1")
        self.model = getattr(settings, "solar_embedding_model", "upstage/solar-embedding-1-large")
        self.use_mock = use_mock or (not settings.should_use_real_embedding)
        
        if self.use_mock:
            logger.warning("Solar Embedding: Using MOCK mode (deterministic)")
        else:
            logger.info("Solar Embedding: Using REAL API")
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if self.use_mock:
            return self._mock_embed(texts)
        
        try:
            return await self._real_embed(texts)
        except httpx.HTTPStatusError as e:
            response_text = ""
            try:
                response_text = e.response.text
            except Exception:
                response_text = ""
            logger.error(
                "Solar Embedding API failed: {} | response={}",
                str(e),
                response_text[:1000]
            )
            return self._mock_embed(texts)
        except Exception as e:
            logger.error(f"Solar Embedding API failed: {e}, falling back to mock")
            return self._mock_embed(texts)
    
    async def embed_single(self, text: str) -> List[float]:
        """Embed single text"""
        embeddings = await self.embed_texts([text])
        return embeddings[0]
    
    async def _real_embed(self, texts: List[str]) -> List[List[float]]:
        """실제 Solar Embedding API 호출"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": texts
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.endpoint}/embeddings",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            embeddings = [item["embedding"] for item in result.get("data", [])]
            
            logger.info(f"Solar Embedding: Generated {len(embeddings)} embeddings")
            return embeddings
    
    def _mock_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Deterministic mock embedding
        - 같은 텍스트 → 같은 embedding
        - 다른 텍스트 → 다른 embedding (해시 기반)
        """
        embeddings = []
        dim = 1024  # Solar embedding dimension
        
        for text in texts:
            # SHA256 해시를 시드로 사용 (deterministic)
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            seed = int(text_hash[:8], 16)
            
            rng = np.random.RandomState(seed)
            embedding = rng.randn(dim).astype(np.float32)
            
            # L2 normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            embeddings.append(embedding.tolist())
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Embedding 차원 반환"""
        return 1024
    
    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        return {
            "provider": "Solar" if not self.use_mock else "Mock",
            "model": "solar-embedding-1-large" if not self.use_mock else "deterministic-hash",
            "dimension": self.get_embedding_dim(),
            "mode": "mock" if self.use_mock else "real"
        }
