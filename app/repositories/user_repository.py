from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Role, User
from app.schemas.user import AgentUpdate
from datetime import datetime, timezone


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

def get_agents_by_company(
    db: Session,
    company_id: int
) -> list[User]:

    statement = (
        select(User)
        .join(Role)
        .where(
            User.company_id == company_id,
            Role.name == "agent"
        )
    )

    return list(
        db.scalars(statement).all()
    )
    
def get_agent_by_id_and_company(
    db: Session,
    agent_id: int,
    company_id: int
) -> User | None:

    statement = (
        select(User)
        .join(Role)
        .where(
            User.id == agent_id,
            User.company_id == company_id,
            Role.name == "agent"
        )
    )

    return db.scalars(statement).first()


def update(
    db: Session,
    user: User,
    data: AgentUpdate
) -> User:

    update_data = data.model_dump(
        exclude_unset=True
    )

    for field, value in update_data.items():
        setattr(
            user,
            field,
            value
        )

    user.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(user)

    return user


def update_status(
    db: Session,
    user: User,
    is_active: bool
) -> User:

    user.is_active = is_active
    user.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(user)

    return user

def get_admins(
    db: Session
) -> list[User]:

    statement = (
        select(User)
        .join(Role)
        .where(
            Role.name == "admin"
        )
    )

    return list(
        db.scalars(statement).all()
    )


def get_admin_by_id(
    db: Session,
    admin_id: int
) -> User | None:

    statement = (
        select(User)
        .join(Role)
        .where(
            User.id == admin_id,
            Role.name == "admin"
        )
    )

    return db.scalars(statement).first()