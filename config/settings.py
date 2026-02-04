# config/settings.py
"""
Configuration for PACS AI Agent
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Solar API (OpenRouter)
    solar_api_key: Optional[str] = Field(default=None, env="SOLAR_API_KEY")
    solar_api_endpoint: str = Field(
        default="https://openrouter.ai/api/v1",
        env="SOLAR_API_ENDPOINT"
    )
    solar_report_model: str = Field(
        default="upstage/solar-pro-3:free",
        env="SOLAR_MODEL"
    )
    solar_embedding_model: str = Field(
        default="upstage/solar-embedding-1-large",
        env="SOLAR_EMBEDDING_MODEL"
    )
    
    # Mock modes
    use_mock_solar: bool = Field(default=False, env="USE_MOCK_SOLAR")
    use_mock_embedding: bool = Field(default=False, env="USE_MOCK_EMBEDDING")
    use_mock_vision: bool = Field(default=True, env="USE_MOCK_VISION")  # Default true (no real models)
    
    # Paths
    dicom_storage_path: Path = Field(default=Path("./data/dicom_storage"))
    dicom_output_path: Path = Field(default=Path("./data/dicom_output"))
    chroma_db_path: Path = Field(default=Path("./data/chroma_db"))
    model_path: Path = Field(default=Path("./models"))
    log_path: Path = Field(default=Path("./logs"))
    
    # API settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # Model settings
    lung_seg_model_path: Optional[Path] = None
    nodule_det_model_path: Optional[Path] = None
    findings_model_path: Optional[Path] = None
    
    # Processing settings
    target_spacing: tuple = (1.0, 1.0, 1.5)  # mm
    roi_size: tuple = (96, 96, 96)
    
    # Threshold settings
    nodule_detection_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    nodule_reporting_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",  # 정의되지 않은 환경변수 무시
        "protected_namespaces": ()  # model_ prefix 경고 제거
    }
    
    def ensure_directories(self):
        """Create necessary directories"""
        self.dicom_storage_path.mkdir(parents=True, exist_ok=True)
        self.dicom_output_path.mkdir(parents=True, exist_ok=True)
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def should_use_real_solar(self) -> bool:
        """Solar API 실제 사용 여부"""
        return bool(self.solar_api_key) and not self.use_mock_solar
    
    @property
    def should_use_real_embedding(self) -> bool:
        """Solar Embedding 실제 사용 여부"""
        return bool(self.solar_api_key) and not self.use_mock_embedding


# Global settings instance
settings = Settings()
settings.ensure_directories()
