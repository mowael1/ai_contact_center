from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.company import Company
from app.schemas.company import CompanyCreate, CompanyUpdate

from datetime import datetime, timezone


def get_all(db: Session) -> list[Company]:
    
    statement = select(Company)
    
    return list(
        db.scalars(statement).all()
    )


def get_by_id(db: Session, company_id: int) -> Company | None:
    
    return db.get(
        Company,
        company_id
    )
    
def get_by_name(db: Session, name: str) -> Company | None:
    
    statement = (
        select(Company).where(Company.name == name)
    )
    
    return db.scalars(statement).first()


def create(
    db: Session,
    data: CompanyCreate
) -> Company:
    
    company = Company(
        name = data.name,
        email = data.email,
        phone = data.phone,
        is_active = data.is_active
    )
    
    # Here we add object to session
    db.add(company)

    # Run SQL code
    db.commit()

    # Here it's reload object from database to get values for (id, created_at)
    db.refresh(company)

    return company

def update(
    db: Session,
    company: Company,
    data: CompanyUpdate
) -> Company:

    update_data = data.model_dump(
        exclude_unset=True
    )

    for field, value in update_data.items():
        setattr(
            company,
            field,
            value
        )

    company.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(company)

    return company

def update_status(
    db: Session,
    company: Company,
    is_active: bool
) -> Company:

    company.is_active = is_active
    company.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(company)

    return company