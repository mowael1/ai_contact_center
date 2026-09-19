from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.dependencies import (
    get_db,
    require_admin,
)
from app.models import User
from app.schemas.customer import (
    CustomerCreate,
    CustomerResponse,
    CustomerStatusUpdate,
    CustomerUpdate,
)
from app.services.customer_service import (
    create_customer,
    get_company_customer_by_id,
    get_company_customers,
    update_company_customer,
    update_company_customer_status,
)


router = APIRouter()

@router.post(
    "",
    response_model=CustomerResponse,
    status_code=status.HTTP_201_CREATED
)
def create_new_customer(
    data: CustomerCreate,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin)
):
    return create_customer(
        db,
        current_admin,
        data
    )
    
@router.get(
    "",
    response_model=list[CustomerResponse]
)
def get_customers(
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin)
):
    return get_company_customers(
        db,
        current_admin
    )
    
@router.get(
    "/{customer_id}",
    response_model=CustomerResponse
)
def get_customer(
    customer_id: int,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin)
):
    return get_company_customer_by_id(
        db,
        current_admin,
        customer_id
    )
    
@router.patch(
    "/{customer_id}",
    response_model=CustomerResponse
)
def update_customer(
    customer_id: int,
    data: CustomerUpdate,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin)
):
    return update_company_customer(
        db,
        current_admin,
        customer_id,
        data
    )
    
@router.patch(
    "/{customer_id}/status",
    response_model=CustomerResponse
)
def change_customer_status(
    customer_id: int,
    data: CustomerStatusUpdate,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin)
):
    return update_company_customer_status(
        db,
        current_admin,
        customer_id,
        data
    )