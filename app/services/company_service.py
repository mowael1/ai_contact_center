from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.company import Company
from app.repositories import company_repository
from app.schemas.company import CompanyCreate, CompanyUpdate,CompanyStatusUpdate


def get_all_companies(db: Session) -> list[Company]:
    return company_repository.get_all(db)


def get_company_by_id(
    db: Session,
    company_id: int
) -> Company | None:

    return company_repository.get_by_id(
        db,
        company_id
    )


def get_company_by_name(
    db: Session,
    name: str
) -> Company | None:

    return company_repository.get_by_name(
        db,
        name
    )
    

def create_company(
    db: Session,
    data: CompanyCreate
) -> Company:

    existing_company = company_repository.get_by_name(
        db,
        data.name
    )

    if existing_company:
        raise HTTPException(
            status_code=409,
            detail="Company already exists"
        )

    return company_repository.create(
        db,
        data
    )
    
def update_company(
    db: Session,
    company_id: int,
    data: CompanyUpdate
) -> Company:

    company = company_repository.get_by_id(
        db,
        company_id
    )

    if company is None:
        raise HTTPException(
            status_code=404,
            detail="Company not found"
        )

    if data.name is not None:
        existing_company = company_repository.get_by_name(
            db,
            data.name
        )

        if (
            existing_company is not None
            and existing_company.id != company_id
        ):
            raise HTTPException(
                status_code=409,
                detail="Company name already exists"
            )

    return company_repository.update(
        db,
        company,
        data
    )
    
def update_company_status(
    db: Session,
    company_id: int,
    data: CompanyStatusUpdate
) -> Company:

    company = company_repository.get_by_id(
        db,
        company_id
    )

    if company is None:
        raise HTTPException(
            status_code=404,
            detail="Company not found"
        )

    return company_repository.update_status(
        db,
        company,
        data.is_active
    )