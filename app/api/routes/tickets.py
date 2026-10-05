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
    TicketUpdate,
)
from app.schemas.analytics import AnalyticsQueryRequest, AnalyticsQueryResponse
from app.services.analytics_service import answer_analytics_question
from app.services.ticket_service import (
    create_ticket,
    get_company_tickets,
    get_my_follow_ups,
    get_ticket_by_id,
    update_ticket,
)

from app.schemas.call import CallResponse
from app.services.call_service import start_ticket_call

router = APIRouter()


@router.post("/analytics/query", response_model=AnalyticsQueryResponse)
def query_ticket_analytics(
    payload: AnalyticsQueryRequest,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin),
):
    """Answer natural-language questions using this admin's company data."""
    history = ""
    if payload.conversation_id and current_admin.company_id is not None:
        from rag.services.chat_store import chat_store

        memory = chat_store.memory(current_admin.company_id, payload.conversation_id)
        recent = "\n".join(
            f"{message['role'].title()}: {message['content']}"
            for message in memory.recent
        )
        history = "\n".join(part for part in (
            f"Conversation summary:\n{memory.summary}" if memory.summary else "",
            f"Recent messages:\n{recent}" if recent else "",
        ) if part)

    result = answer_analytics_question(
        db,
        current_admin,
        payload.query,
        previous_metric=payload.previous_metric,
        previous_time_period=payload.previous_time_period,
        history=history,
    )
    if payload.conversation_id and current_admin.company_id is not None:
        from rag.services.chat_store import chat_store

        chat_store.append(
            current_admin.company_id,
            payload.conversation_id,
            payload.query,
            result["answer"],
            [],
        )
    return result

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


@router.patch(
    "/{ticket_id}",
    response_model=TicketResponse
)
def update_existing_ticket(
    ticket_id: int,
    data: TicketUpdate,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin)
):
    return update_ticket(
        db,
        current_admin,
        ticket_id,
        data
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
