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
    
def get_company_tickets(
    db: Session,
    current_admin: User
) -> list[Ticket]:

    if current_admin.company_id is None:
        raise HTTPException(
            status_code=400,
            detail="Admin is not assigned to a company"
        )

    return ticket_repository.get_by_company(
        db,
        current_admin.company_id
    )
    
def get_my_follow_ups(
    db: Session,
    current_agent: User
) -> list[Ticket]:

    if current_agent.company_id is None:
        raise HTTPException(
            status_code=400,
            detail="Agent is not assigned to a company"
        )

    return ticket_repository.get_agent_follow_ups(
        db,
        company_id=current_agent.company_id,
        agent_id=current_agent.id
    )
    
def get_ticket_by_id(
    db: Session,
    current_user: User,
    ticket_id: int
) -> Ticket:

    if current_user.company_id is None:
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )

    ticket = ticket_repository.get_by_id_and_company(
        db,
        ticket_id,
        current_user.company_id
    )

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="Ticket not found"
        )

    if current_user.role.name == "admin":
        return ticket

    if (
        current_user.role.name == "agent"
        and ticket.assigned_agent_id == current_user.id
    ):
        return ticket

    raise HTTPException(
        status_code=404,
        detail="Ticket not found"
    )