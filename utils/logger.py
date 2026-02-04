# utils/logger.py
"""
Logging utility
"""
import sys
from loguru import logger
from config.settings import settings


def setup_logging():
    """Configure loguru logger"""
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True
    )
    
    # File handler
    log_file = settings.log_path / "pacs_ai_agent_{time:YYYY-MM-DD}.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.log_level,
        rotation="1 day",
        retention="30 days",
        compression="zip"
    )
    
    return logger


# Initialize
setup_logging()
