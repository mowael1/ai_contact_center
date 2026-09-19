from fastapi import APIRouter, Depends,status
from sqlalchemy.orm import Session
from app.models import User

from app.api.dependencies import (
    get_current_user,
    get_db,
    require_admin,
    require_super_admin,
)
from app.schemas.user import (
    AdminUpdate,
    AgentCreate,
    AgentUpdate,
    CurrentUserResponse,
    UserResponse,
    UserStatusUpdate,
)

from app.services.user_service import (
    create_agent,
    get_admin_by_id,
    get_all_admins,
    get_company_agent_by_id,
    get_company_agents,
    update_admin,
    update_admin_status,
    update_company_agent,
    update_company_agent_status,
)

router = APIRouter()

@router.get(
    "/me",
    response_model=CurrentUserResponse
)
def get_my_profile(
    current_user: User = Depends(get_current_user)
):
    return current_user


@router.post(
    "/agents",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED
)
def create_new_agent(
    data: AgentCreate,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin)
):
    return create_agent(
        db,
        current_admin,
        data
    )
    
@router.get(
    "/agents",
    response_model=list[UserResponse]
)
def get_my_company_agents(
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin)
):
    return get_company_agents(
        db,
        current_admin
    )
    
@router.get(
    "/agents/{agent_id}",
    response_model=UserResponse
)
def get_agent(
    agent_id: int,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin)
):
    return get_company_agent_by_id(
        db,
        current_admin,
        agent_id
    )
    
@router.patch(
    "/agents/{agent_id}",
    response_model=UserResponse
)
def update_agent(
    agent_id: int,
    data: AgentUpdate,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin)
):
    return update_company_agent(
        db,
        current_admin,
        agent_id,
        data
    )
    
@router.patch(
    "/agents/{agent_id}/status",
    response_model=UserResponse
)
def change_agent_status(
    agent_id: int,
    data: UserStatusUpdate,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin)
):
    return update_company_agent_status(
        db,
        current_admin,
        agent_id,
        data
    )
    
    
    
# Get all admins
@router.get(
    "/admins",
    response_model=list[UserResponse]
)
def get_admins(
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_super_admin
    )
):
    return get_all_admins(db)

# Get Specific admin
@router.get(
    "/admins/{admin_id}",
    response_model=UserResponse
)
def get_admin(
    admin_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_super_admin
    )
):
    return get_admin_by_id(
        db,
        admin_id
    )
    
# Edit Admin
@router.patch(
    "/admins/{admin_id}",
    response_model=UserResponse
)
def update_existing_admin(
    admin_id: int,
    data: AdminUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_super_admin
    )
):
    return update_admin(
        db,
        admin_id,
        data
    )
    
# Activate/Deactivate Admin
@router.patch(
    "/admins/{admin_id}/status",
    response_model=UserResponse
)
def change_admin_status(
    admin_id: int,
    data: UserStatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_super_admin
    )
):
    return update_admin_status(
        db,
        admin_id,
        data
    )