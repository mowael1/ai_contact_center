from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import (
    get_db,
    require_super_admin,
)

from app.models import User

from app.schemas.company import (
    CompanyCreate,
    CompanyResponse,
    CompanyStatusUpdate,
    CompanyUpdate,
)
from app.services.company_service import (
    create_company,
    get_all_companies,
    get_company_by_id,
    update_company,
    update_company_status,
)

from app.schemas.user import (
    CompanyAdminCreate,
    UserResponse,
)

from app.services.user_service import (
    create_company_admin,
)

router = APIRouter(
    dependencies=[
        Depends(require_super_admin)
    ]
)

@router.get(
    "/",
    response_model=list[CompanyResponse]
)
def get_companies(
    db: Session = Depends(get_db)
):
    return get_all_companies(db)


@router.get(
    "/{company_id}",
    response_model=CompanyResponse
)
def get_company(
    company_id: int,
    db: Session = Depends(get_db)
):
    company = get_company_by_id(
        db,
        company_id
    )

    if company is None:
        raise HTTPException(
            status_code=404,
            detail="Company not found"
        )

    return company


@router.post(
    "/",
    response_model=CompanyResponse,
    status_code=status.HTTP_201_CREATED
)
def create_new_company(
    data: CompanyCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_super_admin
    )
):
    return create_company(
        db,
        data
    )
    
@router.patch(
    "/{company_id}",
    response_model=CompanyResponse
)
def update_existing_company(
    company_id: int,
    data: CompanyUpdate,
    db: Session = Depends(get_db)
):
    return update_company(
        db,
        company_id,
        data
    )
    
@router.patch(
    "/{company_id}/status",
    response_model=CompanyResponse
)
def change_company_status(
    company_id: int,
    data: CompanyStatusUpdate,
    db: Session = Depends(get_db)
):
    return update_company_status(
        db,
        company_id,
        data
    )
    
@router.post(
    "/{company_id}/admins",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED
)
def create_admin_for_company(
    company_id: int,
    data: CompanyAdminCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_super_admin
    )
):
    return create_company_admin(
        db,
        company_id,
        data
    )