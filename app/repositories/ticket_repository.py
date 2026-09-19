from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

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

def get_by_company(
    db: Session,
    company_id: int
) -> list[Ticket]:

    statement = (
        select(Ticket)
        .where(
            Ticket.company_id == company_id
        )
        .order_by(
            Ticket.created_at.desc()
        )
    )

    return list(
        db.scalars(statement).all()
    )
    
def get_by_id_and_company(
    db: Session,
    ticket_id: int,
    company_id: int
) -> Ticket | None:

    statement = (
        select(Ticket)
        .where(
            Ticket.id == ticket_id,
            Ticket.company_id == company_id
        )
    )

    return db.scalars(statement).first()

def get_agent_follow_ups(
    db: Session,
    *,
    company_id: int,
    agent_id: int
) -> list[Ticket]:

    statement = (
        select(Ticket)
        .options(
            selectinload(Ticket.customer)
        )
        .where(
            Ticket.company_id == company_id,
            Ticket.assigned_agent_id == agent_id,
            Ticket.status == "pending_follow_up"
        )
        .order_by(
            Ticket.created_at.asc()
        )
    )

    return list(
        db.scalars(statement).all()
    )