from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.dependencies import (
    get_current_user,
    get_db,
    require_admin,
    require_agent,
)
from app.models import User
from app.schemas.ticket import (
    FollowUpTicketResponse,
    TicketCreate,
    TicketResponse,
)
from app.services.ticket_service import (
    create_ticket,
    get_company_tickets,
    get_my_follow_ups,
    get_ticket_by_id,
)

from app.schemas.call import CallResponse
from app.services.call_service import start_ticket_call

router = APIRouter()

@router.post(
    "",
    response_model=TicketResponse,
    status_code=status.HTTP_201_CREATED
)
def create_new_ticket(
    data: TicketCreate,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin)
):
    return create_ticket(
        db,
        current_admin,
        data
    )

# Only admin can see all company tickets
@router.get(
    "",
    response_model=list[TicketResponse]
)
def get_tickets(
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin)
):
    return get_company_tickets(
        db,
        current_admin
    )

@router.get(
    "/my-follow-ups",
    response_model=list[FollowUpTicketResponse]
)
def get_agent_follow_ups(
    db: Session = Depends(get_db),
    current_agent: User = Depends(require_agent)
):
    return get_my_follow_ups(
        db,
        current_agent
    )

@router.get(
    "/{ticket_id}",
    response_model=TicketResponse
)
def get_ticket(
    ticket_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return get_ticket_by_id(
        db,
        current_user,
        ticket_id
    )
    
@router.post(
    "/{ticket_id}/call",
    response_model=CallResponse,
    status_code=status.HTTP_201_CREATED
)
def start_call(
    ticket_id: int,
    db: Session = Depends(get_db),
    current_agent: User = Depends(require_agent)
):
    return start_ticket_call(
        db,
        current_agent,
        ticket_id
    )