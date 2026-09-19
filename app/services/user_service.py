from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models import User
from app.repositories import (
    company_repository,
    role_repository,
    user_repository,
)
from app.schemas.user import CompanyAdminCreate,AgentCreate

from app.schemas.user import (
    AgentUpdate,
    UserStatusUpdate,
    AdminUpdate
)

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
    
def create_agent(
    db: Session,
    current_admin: User,
    data: AgentCreate
) -> User:

    if current_admin.company_id is None:
        raise HTTPException(
            status_code=400,
            detail="Admin is not assigned to a company"
        )

    existing_user = user_repository.get_by_email(
        db,
        data.email
    )

    if existing_user is not None:
        raise HTTPException(
            status_code=409,
            detail="Email already exists"
        )

    agent_role = role_repository.get_by_name(
        db,
        "agent"
    )

    if agent_role is None:
        raise HTTPException(
            status_code=500,
            detail="Agent role not configured"
        )

    hashed_password = hash_password(
        data.password
    )

    return user_repository.create(
        db,
        full_name=data.full_name,
        email=data.email,
        password_hash=hashed_password,
        role_id=agent_role.id,
        company_id=current_admin.company_id,
        is_active=data.is_active
    )
    
def get_company_agents(
    db: Session,
    current_admin: User
) -> list[User]:

    if current_admin.company_id is None:
        raise HTTPException(
            status_code=400,
            detail="Admin is not assigned to a company"
        )

    return user_repository.get_agents_by_company(
        db,
        current_admin.company_id
    )
    
def get_company_agent_by_id(
    db: Session,
    current_admin: User,
    agent_id: int
) -> User:

    if current_admin.company_id is None:
        raise HTTPException(
            status_code=400,
            detail="Admin is not assigned to a company"
        )

    agent = user_repository.get_agent_by_id_and_company(
        db,
        agent_id,
        current_admin.company_id
    )

    if agent is None:
        raise HTTPException(
            status_code=404,
            detail="Agent not found"
        )

    return agent

def update_company_agent(
    db: Session,
    current_admin: User,
    agent_id: int,
    data: AgentUpdate
) -> User:

    if current_admin.company_id is None:
        raise HTTPException(
            status_code=400,
            detail="Admin is not assigned to a company"
        )

    agent = user_repository.get_agent_by_id_and_company(
        db,
        agent_id,
        current_admin.company_id
    )

    if agent is None:
        raise HTTPException(
            status_code=404,
            detail="Agent not found"
        )

    if data.email is not None:

        existing_user = user_repository.get_by_email(
            db,
            data.email
        )

        if (
            existing_user is not None
            and existing_user.id != agent.id
        ):
            raise HTTPException(
                status_code=409,
                detail="Email already exists"
            )

    return user_repository.update(
        db,
        agent,
        data
    )
    
def update_company_agent_status(
    db: Session,
    current_admin: User,
    agent_id: int,
    data: UserStatusUpdate
) -> User:

    if current_admin.company_id is None:
        raise HTTPException(
            status_code=400,
            detail="Admin is not assigned to a company"
        )

    agent = user_repository.get_agent_by_id_and_company(
        db,
        agent_id,
        current_admin.company_id
    )

    if agent is None:
        raise HTTPException(
            status_code=404,
            detail="Agent not found"
        )

    return user_repository.update_status(
        db,
        agent,
        data.is_active
    )
    

def get_all_admins(
    db: Session
) -> list[User]:

    return user_repository.get_admins(db)


def get_admin_by_id(
    db: Session,
    admin_id: int
) -> User:

    admin = user_repository.get_admin_by_id(
        db,
        admin_id
    )

    if admin is None:
        raise HTTPException(
            status_code=404,
            detail="Admin not found"
        )

    return admin


def update_admin(
    db: Session,
    admin_id: int,
    data: AdminUpdate
) -> User:

    admin = user_repository.get_admin_by_id(
        db,
        admin_id
    )

    if admin is None:
        raise HTTPException(
            status_code=404,
            detail="Admin not found"
        )

    if data.email is not None:

        existing_user = user_repository.get_by_email(
            db,
            data.email
        )

        if (
            existing_user is not None
            and existing_user.id != admin.id
        ):
            raise HTTPException(
                status_code=409,
                detail="Email already exists"
            )

    return user_repository.update(
        db,
        admin,
        data
    )
    
def update_admin_status(
    db: Session,
    admin_id: int,
    data: UserStatusUpdate
) -> User:

    admin = user_repository.get_admin_by_id(
        db,
        admin_id
    )

    if admin is None:
        raise HTTPException(
            status_code=404,
            detail="Admin not found"
        )

    return user_repository.update_status(
        db,
        admin,
        data.is_active
    )