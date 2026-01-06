"""
JWT Authentication Handler for ShuttleSense
Provides token creation, verification, and FastAPI dependency injection.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from fastapi import Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError, ExpiredSignatureError
from passlib.context import CryptContext
from pydantic import BaseModel

from config.settings import get_settings
from exceptions import (
    AuthenticationError,
    TokenExpiredError,
    InvalidTokenError
)

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer security scheme
security = HTTPBearer(auto_error=False)


class TokenPayload(BaseModel):
    """JWT Token payload structure"""
    sub: str  # Subject (user_id)
    exp: datetime  # Expiration time
    iat: datetime  # Issued at
    type: str = "access"  # Token type: access or refresh


class UserInfo(BaseModel):
    """Authenticated user information"""
    user_id: str
    email: Optional[str] = None
    is_active: bool = True


class JWTHandler:
    """
    Handles JWT token creation and verification.
    Supports access tokens and refresh tokens.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.secret_key = self.settings.JWT_SECRET_KEY
        self.algorithm = self.settings.JWT_ALGORITHM
        self.access_token_expire_hours = self.settings.JWT_ACCESS_TOKEN_EXPIRE_HOURS
        self.refresh_token_expire_days = self.settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
    
    def create_access_token(
        self,
        user_id: str,
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new access token.
        
        Args:
            user_id: The user's unique identifier
            additional_claims: Optional additional JWT claims
        
        Returns:
            Encoded JWT access token
        """
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=self.access_token_expire_hours)
        
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": expires,
            "type": "access"
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """
        Create a new refresh token.
        
        Args:
            user_id: The user's unique identifier
        
        Returns:
            Encoded JWT refresh token
        """
        now = datetime.now(timezone.utc)
        expires = now + timedelta(days=self.refresh_token_expire_days)
        
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": expires,
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str, expected_type: str = "access") -> str:
        """
        Verify and decode a JWT token.
        
        Args:
            token: The JWT token to verify
            expected_type: Expected token type (access or refresh)
        
        Returns:
            The user_id from the token
        
        Raises:
            TokenExpiredError: If token has expired
            InvalidTokenError: If token is invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            user_id = payload.get("sub")
            token_type = payload.get("type", "access")
            
            if not user_id:
                raise InvalidTokenError("Token missing user ID")
            
            if token_type != expected_type:
                raise InvalidTokenError(f"Expected {expected_type} token, got {token_type}")
            
            return user_id
            
        except ExpiredSignatureError:
            logger.warning("Token expired")
            raise TokenExpiredError()
        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            raise InvalidTokenError(str(e))
    
    def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """
        Use a refresh token to get a new access token.
        
        Args:
            refresh_token: Valid refresh token
        
        Returns:
            Dict with new access_token and refresh_token
        """
        user_id = self.verify_token(refresh_token, expected_type="refresh")
        
        return {
            "access_token": self.create_access_token(user_id),
            "refresh_token": self.create_refresh_token(user_id),
            "token_type": "bearer"
        }
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)


# Global JWT handler instance
jwt_handler = JWTHandler()


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> UserInfo:
    """
    FastAPI dependency to get current authenticated user.
    Raises AuthenticationError if not authenticated.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: UserInfo = Depends(get_current_user)):
            return {"user_id": user.user_id}
    """
    if not credentials:
        raise AuthenticationError("No authentication credentials provided")
    
    user_id = jwt_handler.verify_token(credentials.credentials)
    
    return UserInfo(user_id=user_id)


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[UserInfo]:
    """
    FastAPI dependency to optionally get current user.
    Returns None if not authenticated (doesn't raise error).
    
    Usage:
        @app.get("/public-with-optional-auth")
        async def route(user: Optional[UserInfo] = Depends(get_current_user_optional)):
            if user:
                return {"user_id": user.user_id}
            return {"message": "Anonymous access"}
    """
    if not credentials:
        return None
    
    try:
        user_id = jwt_handler.verify_token(credentials.credentials)
        return UserInfo(user_id=user_id)
    except Exception:
        return None


def get_user_id_from_request(request: Request) -> Optional[str]:
    """
    Extract user ID from request headers if present.
    Utility function for logging/middleware.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    
    token = auth_header.split(" ")[1]
    try:
        return jwt_handler.verify_token(token)
    except Exception:
        return None
