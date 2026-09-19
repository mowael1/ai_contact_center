from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models import Ticket, User
from app.repositories import (
    customer_repository,
    ticket_repository,
    user_repository,
)
from app.schemas.ticket import TicketCreate


def create_ticket(
    db: Session,
    current_admin: User,
    data: TicketCreate
) -> Ticket:

    if current_admin.company_id is None:
        raise HTTPException(
            status_code=400,
            detail="Admin is not assigned to a company"
        )

    customer = customer_repository.get_by_id_and_company(
        db,
        data.customer_id,
        current_admin.company_id
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

    agent = user_repository.get_agent_by_id_and_company(
        db,
        data.assigned_agent_id,
        current_admin.company_id
    )

    if agent is None:
        raise HTTPException(
            status_code=404,
            detail="Agent not found"
        )

    if not agent.is_active:
        raise HTTPException(
            status_code=400,
            detail="Agent is inactive"
        )

    return ticket_repository.create(
        db,
        company_id=current_admin.company_id,
        customer_id=customer.id,
        assigned_agent_id=agent.id,
        subject=data.subject,
        description=data.description,
        procedure_steps=data.procedure_steps
    )