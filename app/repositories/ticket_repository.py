from sqlalchemy.orm import Session
from app.models import Ticket


def create(
    db: Session,
    *,
    company_id: int,
    customer_id: int,
    assigned_agent_id: int,
    subject: str,
    description: str | None,
    procedure_steps: str | None
) -> Ticket:

    ticket = Ticket(
        company_id=company_id,
        customer_id=customer_id,
        assigned_agent_id=assigned_agent_id,
        subject=subject,
        description=description,
        procedure_steps=procedure_steps,
        status="pending_follow_up"
    )

    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    return ticket