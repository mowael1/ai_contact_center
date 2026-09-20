from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Call


def create(
    db: Session,
    *,
    company_id: int,
    ticket_id: int,
    customer_id: int,
    agent_id: int
) -> Call:

    now = datetime.now(
        timezone.utc
    ).replace(tzinfo=None)

    call = Call(
        company_id=company_id,
        ticket_id=ticket_id,
        customer_id=customer_id,
        agent_id=agent_id,
        status="initiating",
        started_at=now
    )

    db.add(call)
    db.commit()
    db.refresh(call)

    return call

def get_by_id_and_agent(
    db: Session,
    *,
    call_id: int,
    company_id: int,
    agent_id: int
) -> Call | None:

    statement = (
        select(Call)
        .where(
            Call.id == call_id,
            Call.company_id == company_id,
            Call.agent_id == agent_id
        )
    )

    return db.scalars(
        statement
    ).first()
    
def get_active_by_ticket(
    db: Session,
    ticket_id: int
) -> Call | None:

    statement = (
        select(Call)
        .where(
            Call.ticket_id == ticket_id,
            Call.status.in_(
                [
                    "initiating",
                    "ringing",
                    "in_progress",
                ]
            )
        )
    )

    return db.scalars(
        statement
    ).first()