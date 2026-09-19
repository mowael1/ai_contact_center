from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import (
    create_access_token,
    verify_password,
)
from app.repositories import user_repository
from app.schemas.auth import (
    LoginRequest,
    TokenResponse,
)


def login(
    db: Session,
    data: LoginRequest
) -> TokenResponse:

    user = user_repository.get_by_email(
        db,
        data.email
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if not verify_password(
        data.password,
        user.password_hash
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is inactive"
        )
        
    if (
        user.company is not None
        and not user.company.is_active
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Company is inactive"
        )

    access_token = create_access_token(
        user.id
    )

    return TokenResponse(
        access_token=access_token
    )