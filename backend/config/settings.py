"""
Centralized Settings Management using Pydantic Settings
Secure configuration with environment variable support.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Use .env file for local development, cloud secrets manager for production.
    """
    
    # Application
    APP_NAME: str = "ShuttleSense Video Coach API"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    ENVIRONMENT: str = Field(default="development", description="Environment: development, staging, production")
    
    # Security
    JWT_SECRET_KEY: str = Field(default="CHANGE_ME_IN_PRODUCTION", description="Secret key for JWT signing")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_HOURS: int = 24
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # CORS
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Comma-separated list of allowed origins"
    )
    CORS_ALLOW_CREDENTIALS: bool = False
    
    # API Keys
    GOOGLE_API_KEY: Optional[str] = Field(default=None, description="Google Gemini API key for chat")
    
    # LLM Settings
    LLM_PROVIDER: str = Field(default="gemini", description="gemini or anthropic (for proxy)")
    LLM_PROXY_URL: Optional[str] = Field(default="http://localhost:8000/v1", description="URL for LLM proxy")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_UPLOAD: str = "10/hour"
    RATE_LIMIT_ANALYZE: str = "5/hour"
    RATE_LIMIT_CHAT: str = "30/hour"
    RATE_LIMIT_GLOBAL: str = "100/hour"
    
    # File Upload
    MAX_VIDEO_SIZE_MB: int = 500
    MAX_UPLOAD_DIR_SIZE_GB: int = 5
    UPLOAD_DIR: str = "./data/uploads"
    SESSIONS_DIR: str = "./data/sessions"
    
    # Database (for future migration)
    DATABASE_URL: Optional[str] = Field(default=None, description="Database connection URL")
    
    @property
    def allowed_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string"""
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]
    
    @property
    def max_video_size_bytes(self) -> int:
        """Get max video size in bytes"""
        return self.MAX_VIDEO_SIZE_MB * 1024 * 1024
    
    @property
    def max_upload_dir_size_bytes(self) -> int:
        """Get max upload directory size in bytes"""
        return self.MAX_UPLOAD_DIR_SIZE_GB * 1024 ** 3
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENVIRONMENT.lower() == "development"
    
    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def validate_jwt_secret(cls, v: str, info) -> str:
        """Warn if using default secret in non-development"""
        if v == "CHANGE_ME_IN_PRODUCTION":
            env = os.getenv("ENVIRONMENT", "development")
            if env.lower() != "development":
                raise ValueError("JWT_SECRET_KEY must be set in production!")
        return v
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore"
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache for performance - settings are loaded once.
    """
    return Settings()


# Convenience function for direct access
settings = get_settings()
