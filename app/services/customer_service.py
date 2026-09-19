from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models import Customer, User
from app.repositories import customer_repository
from app.schemas.customer import (
    CustomerCreate,
    CustomerStatusUpdate,
    CustomerUpdate,
)


def create_customer(
    db: Session,
    current_admin: User,
    data: CustomerCreate
) -> Customer:

    if current_admin.company_id is None:
        raise HTTPException(
            status_code=400,
            detail="Admin is not assigned to a company"
        )

    return customer_repository.create(
        db,
        company_id=current_admin.company_id,
        full_name=data.full_name,
        phone=data.phone,
        email=data.email,
        is_active=data.is_active
    )
    
def get_company_customers(
    db: Session,
    current_admin: User
) -> list[Customer]:

    if current_admin.company_id is None:
        raise HTTPException(
            status_code=400,
            detail="Admin is not assigned to a company"
        )

    return customer_repository.get_by_company(
        db,
        current_admin.company_id
    )
    
def get_company_customer_by_id(
    db: Session,
    current_admin: User,
    customer_id: int
) -> Customer:

    if current_admin.company_id is None:
        raise HTTPException(
            status_code=400,
            detail="Admin is not assigned to a company"
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

    return customer

def update_company_customer(
    db: Session,
    current_admin: User,
    customer_id: int,
    data: CustomerUpdate
) -> Customer:

    if current_admin.company_id is None:
        raise HTTPException(
            status_code=400,
            detail="Admin is not assigned to a company"
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

    return customer_repository.update(
        db,
        customer,
        data
    )
    
def update_company_customer_status(
    db: Session,
    current_admin: User,
    customer_id: int,
    data: CustomerStatusUpdate
) -> Customer:

    if current_admin.company_id is None:
        raise HTTPException(
            status_code=400,
            detail="Admin is not assigned to a company"
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

    return customer_repository.update_status(
        db,
        customer,
        data.is_active
    )