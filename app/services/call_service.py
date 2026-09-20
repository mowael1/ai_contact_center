from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.integrations.vonage_client import (
    create_outbound_call,
)
from app.models import Call, User
from app.repositories import (
    call_repository,
    customer_repository,
    ticket_repository,
)
from app.schemas.call import MockCallResult


def _utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    ).replace(tzinfo=None)


# =========================================
# Start real outbound call
# =========================================

def start_ticket_call(
    db: Session,
    current_agent: User,
    ticket_id: int
) -> Call:

    # =====================================
    # 1. Agent must belong to a company
    # =====================================

    if current_agent.company_id is None:
        raise HTTPException(
            status_code=400,
            detail="Agent is not assigned to a company"
        )

    # =====================================
    # 2. Get ticket inside agent company
    # =====================================

    ticket = (
        ticket_repository.get_by_id_and_company(
            db,
            ticket_id,
            current_agent.company_id
        )
    )

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="Ticket not found"
        )

    # =====================================
    # 3. Ticket must belong to this agent
    # =====================================

    if ticket.assigned_agent_id != current_agent.id:
        raise HTTPException(
            status_code=404,
            detail="Ticket not found"
        )

    # =====================================
    # 4. Ticket must need follow-up
    # =====================================

    if ticket.status != "pending_follow_up":
        raise HTTPException(
            status_code=409,
            detail="Ticket is not pending follow-up"
        )

    # =====================================
    # 5. Get customer
    # =====================================

    customer = (
        customer_repository.get_by_id_and_company(
            db,
            ticket.customer_id,
            current_agent.company_id
        )
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

    # =====================================
    # 6. Customer must have phone number
    # =====================================

    if not customer.phone:
        raise HTTPException(
            status_code=400,
            detail="Customer does not have a phone number"
        )

    # =====================================
    # 7. Prevent duplicate active calls
    # =====================================

    active_call = (
        call_repository.get_active_by_ticket(
            db,
            ticket.id
        )
    )

    if active_call is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "There is already an active "
                "call for this ticket"
            )
        )

    # =====================================
    # 8. Create local Call first
    # =====================================

    call = call_repository.create(
        db,
        company_id=current_agent.company_id,
        ticket_id=ticket.id,
        customer_id=customer.id,
        agent_id=current_agent.id
    )

    print("\n==============================")
    print("LOCAL CALL CREATED")
    print("CALL ID:", call.id)
    print("TICKET ID:", ticket.id)
    print("CUSTOMER ID:", customer.id)
    print("CUSTOMER:", customer.full_name)
    print("PHONE:", customer.phone)
    print("STATUS:", call.status)
    print("==============================\n")

    # =====================================
    # 9. Start real Vonage call
    # =====================================

    try:

        provider_call_id = create_outbound_call(
            phone_number=customer.phone
        )

    except Exception as exc:

        print("\n==============================")
        print("VONAGE CALL FAILED")
        print("LOCAL CALL ID:", call.id)
        print("ERROR:", str(exc))
        print("==============================\n")

        now = _utc_now()

        call.status = "failed"
        call.outcome = None
        call.ended_at = now

        if call.started_at is not None:
            call.duration_seconds = max(
                0,
                int(
                    (
                        now - call.started_at
                    ).total_seconds()
                )
            )

        # Ticket still needs follow-up
        ticket.status = "pending_follow_up"
        ticket.updated_at = now

        db.commit()
        db.refresh(call)

        raise HTTPException(
            status_code=502,
            detail="Failed to start outbound call"
        )

    # =====================================
    # 10. Save Vonage UUID
    # =====================================

    call.provider_call_id = provider_call_id

    db.commit()
    db.refresh(call)

    print("\n==============================")
    print("VONAGE CALL LINKED")
    print("LOCAL CALL ID:", call.id)
    print(
        "PROVIDER CALL ID:",
        call.provider_call_id
    )
    print("STATUS:", call.status)
    print("==============================\n")

    return call


# =========================================
# Mock call result
# =========================================

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

    call = (
        call_repository.get_by_id_and_agent(
            db,
            call_id=call_id,
            company_id=current_agent.company_id,
            agent_id=current_agent.id
        )
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

    ticket = (
        ticket_repository.get_by_id_and_company(
            db,
            call.ticket_id,
            current_agent.company_id
        )
    )

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="Ticket not found"
        )

    now = _utc_now()

    call.ended_at = now

    if call.started_at is not None:
        call.duration_seconds = int(
            (
                now - call.started_at
            ).total_seconds()
        )

    # =====================================
    # Resolved
    # =====================================

    if data.result == "resolved":

        call.status = "completed"
        call.outcome = "resolved"

        ticket.status = "resolved"

    # =====================================
    # Not resolved
    # =====================================

    elif data.result == "not_resolved":

        call.status = "completed"
        call.outcome = "not_resolved"

        ticket.status = "needs_agent"

    # =====================================
    # Unclear
    # =====================================

    elif data.result == "unclear":

        call.status = "completed"
        call.outcome = "unclear"

        ticket.status = "needs_agent"

    # =====================================
    # No answer
    # =====================================

    elif data.result == "no_answer":

        call.status = "no_answer"
        call.outcome = None

        ticket.status = "pending_follow_up"

    # =====================================
    # Failed
    # =====================================

    elif data.result == "failed":

        call.status = "failed"
        call.outcome = None

        ticket.status = "pending_follow_up"

    ticket.updated_at = now

    db.commit()
    db.refresh(call)

    return call