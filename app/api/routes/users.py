from fastapi import APIRouter, Depends

from app.api.dependencies import get_current_user
from app.models import User
from app.schemas.user import CurrentUserResponse


router = APIRouter()


@router.get(
    "/me",
    response_model=CurrentUserResponse
)
def get_my_profile(
    current_user: User = Depends(get_current_user)
):
    return current_user