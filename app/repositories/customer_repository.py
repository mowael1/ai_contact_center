from datetime import datetime, timezone

from sqlalchemy.orm import Session
from app.models import Customer
from sqlalchemy import select
from app.schemas.customer import CustomerUpdate


def create(
    db: Session,
    *,
    company_id: int,
    full_name: str,
    phone: str,
    email: str | None,
    is_active: bool = True
) -> Customer:

    customer = Customer(
        company_id=company_id,
        full_name=full_name,
        phone=phone,
        email=email,
        is_active=is_active
    )

    db.add(customer)
    db.commit()
    db.refresh(customer)

    return customer


def get_by_company(
    db: Session,
    company_id: int
) -> list[Customer]:

    statement = (
        select(Customer)
        .where(
            Customer.company_id == company_id
        )
    )

    return list(
        db.scalars(statement).all()
    )


def get_by_id_and_company(
    db: Session,
    customer_id: int,
    company_id: int
) -> Customer | None:

    statement = (
        select(Customer)
        .where(
            Customer.id == customer_id,
            Customer.company_id == company_id
        )
    )

    return db.scalars(statement).first()


def update(
    db: Session,
    customer: Customer,
    data: CustomerUpdate
) -> Customer:

    update_data = data.model_dump(
        exclude_unset=True
    )

    for field, value in update_data.items():
        setattr(
            customer,
            field,
            value
        )

    customer.updated_at = datetime.now(
        timezone.utc
    )

    db.commit()
    db.refresh(customer)

    return customer


def update_status(
    db: Session,
    customer: Customer,
    is_active: bool
) -> Customer:

    customer.is_active = is_active

    customer.updated_at = datetime.now(
        timezone.utc
    )

    db.commit()
    db.refresh(customer)

    return customer