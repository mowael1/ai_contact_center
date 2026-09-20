from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models import Call, User
from app.repositories import (
    call_repository,
    customer_repository,
    ticket_repository,
)

from datetime import datetime, timezone

from app.schemas.call import MockCallResult

def start_ticket_call(
    db: Session,
    current_agent: User,
    ticket_id: int
) -> Call:

    if current_agent.company_id is None:
        raise HTTPException(
            status_code=400,
            detail="Agent is not assigned to a company"
        )

    ticket = ticket_repository.get_by_id_and_company(
        db,
        ticket_id,
        current_agent.company_id
    )

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="Ticket not found"
        )

    if ticket.assigned_agent_id != current_agent.id:
        raise HTTPException(
            status_code=404,
            detail="Ticket not found"
        )

    if ticket.status != "pending_follow_up":
        raise HTTPException(
            status_code=409,
            detail="Ticket is not pending follow-up"
        )

    customer = customer_repository.get_by_id_and_company(
        db,
        ticket.customer_id,
        current_agent.company_id
    )

    if customer is None:
        raise HTTPException(
            status_code=404,
            detail="Customer not found"
        )

    if not customer.is_active:
        raise HTTPException(
            status_code=400,
            detail="Customer is inactive"
        )

    active_call = call_repository.get_active_by_ticket(
        db,
        ticket.id
    )

    if active_call is not None:
        raise HTTPException(
            status_code=409,
            detail="There is already an active call for this ticket"
        )

    return call_repository.create(
        db,
        company_id=current_agent.company_id,
        ticket_id=ticket.id,
        customer_id=customer.id,
        agent_id=current_agent.id
    )
    
    
# Mock
def set_mock_call_result(
    db: Session,
    current_agent: User,
    call_id: int,
    data: MockCallResult
) -> Call:

    if current_agent.company_id is None:
        raise HTTPException(
            status_code=400,
            detail="Agent is not assigned to a company"
        )

    call = call_repository.get_by_id_and_agent(
        db,
        call_id=call_id,
        company_id=current_agent.company_id,
        agent_id=current_agent.id
    )

    if call is None:
        raise HTTPException(
            status_code=404,
            detail="Call not found"
        )

    if call.status in {
        "completed",
        "no_answer",
        "failed",
    }:
        raise HTTPException(
            status_code=409,
            detail="Call is already finished"
        )

    ticket = ticket_repository.get_by_id_and_company(
        db,
        call.ticket_id,
        current_agent.company_id
    )

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="Ticket not found"
        )

    now = datetime.now(
        timezone.utc
    ).replace(tzinfo=None)

    call.ended_at = now

    if call.started_at is not None:
        call.duration_seconds = int(
            (
                now - call.started_at
            ).total_seconds()
        )

    if data.result == "resolved":

        call.status = "completed"
        call.outcome = "resolved"

        ticket.status = "resolved"

    elif data.result == "not_resolved":

        call.status = "completed"
        call.outcome = "not_resolved"

        ticket.status = "needs_agent"

    elif data.result == "unclear":

        call.status = "completed"
        call.outcome = "unclear"

        ticket.status = "needs_agent"

    elif data.result == "no_answer":

        call.status = "no_answer"
        call.outcome = None

        ticket.status = "pending_follow_up"

    elif data.result == "failed":

        call.status = "failed"
        call.outcome = None

        ticket.status = "pending_follow_up"

    ticket.updated_at = now

    db.commit()
    db.refresh(call)

    return call