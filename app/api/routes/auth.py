from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.schemas.auth import(
    LoginRequest,
    TokenResponse
)

from app.services.auth_service import login

router = APIRouter()

@router.post(
    "/login",
    response_model=TokenResponse
)
def login_user(
    data: LoginRequest,
    db: Session = Depends(get_db)
):
    
    return login(db,data)