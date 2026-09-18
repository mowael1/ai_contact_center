from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import User

def get_all(db: Session) -> list[User]:
    statement = select(User)

    return list(
        db.scalars(statement).all()
    )

def get_by_id(
    db: Session,
    user_id: int
) -> User | None:

    return db.get(
        User,
        user_id
    )

def get_by_email(
    db: Session,
    email: str
) -> User | None:

    statement = (
        select(User)
        .where(User.email == email)
    )

    return db.scalars(statement).first()

def create(
    db: Session,
    *,
    full_name: str,
    email: str,
    password_hash: str,
    role_id: int,
    company_id: int | None,
    is_active: bool = True
) -> User:

    user = User(
        full_name=full_name,
        email=email,
        password_hash=password_hash,
        role_id=role_id,
        company_id=company_id,
        is_active=is_active
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user