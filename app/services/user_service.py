from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models import User
from app.repositories import (
    company_repository,
    role_repository,
    user_repository,
)
from app.schemas.user import CompanyAdminCreate


def create_company_admin(
    db: Session,
    company_id: int,
    data: CompanyAdminCreate
) -> User:

    company = company_repository.get_by_id(
        db,
        company_id
    )

    if company is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found"
        )

    existing_user = user_repository.get_by_email(
        db,
        data.email
    )

    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already exists"
        )

    admin_role = role_repository.get_by_name(
        db,
        "admin"
    )

    if admin_role is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin role not configured"
        )

    hashed_password = hash_password(
        data.password
    )

    return user_repository.create(
        db,
        full_name=data.full_name,
        email=data.email,
        password_hash=hashed_password,
        role_id=admin_role.id,
        company_id=company.id,
        is_active=data.is_active
    )