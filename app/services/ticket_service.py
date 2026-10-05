from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models import Ticket, User
from app.repositories import (
    customer_repository,
    ticket_repository,
    user_repository,
)
from app.schemas.ticket import TicketCreate, TicketUpdate


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


def update_ticket(
    db: Session,
    current_admin: User,
    ticket_id: int,
    data: TicketUpdate,
) -> Ticket:

    if current_admin.company_id is None:
        raise HTTPException(
            status_code=400,
            detail="Admin is not assigned to a company"
        )

    ticket = ticket_repository.get_by_id_and_company(
        db,
        ticket_id,
        current_admin.company_id
    )

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="Ticket not found"
        )

    update_data = data.model_dump(
        exclude_unset=True
    )

    if not update_data:
        return ticket

    if "subject" in update_data:
        subject = update_data["subject"]

        if subject is None or not subject.strip():
            raise HTTPException(
                status_code=422,
                detail="Ticket subject cannot be empty"
            )

        data.subject = subject.strip()

    if "customer_id" in update_data:
        customer_id = update_data["customer_id"]

        if customer_id is None:
            raise HTTPException(
                status_code=422,
                detail="Customer is required"
            )

        customer = customer_repository.get_by_id_and_company(
            db,
            customer_id,
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

    if "assigned_agent_id" in update_data:
        assigned_agent_id = update_data["assigned_agent_id"]

        if assigned_agent_id is None:
            raise HTTPException(
                status_code=422,
                detail="Assigned agent is required"
            )

        agent = user_repository.get_agent_by_id_and_company(
            db,
            assigned_agent_id,
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

    if "status" in update_data and update_data["status"] is None:
        raise HTTPException(
            status_code=422,
            detail="Ticket status is required"
        )

    return ticket_repository.update(
        db,
        ticket,
        data
    )
