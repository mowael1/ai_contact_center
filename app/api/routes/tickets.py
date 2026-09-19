from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.dependencies import (
    get_db,
    require_admin,
)
from app.models import User
from app.schemas.ticket import (
    TicketCreate,
    TicketResponse,
)
from app.services.ticket_service import (
    create_ticket,
)

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