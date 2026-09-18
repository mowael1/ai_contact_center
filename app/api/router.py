from fastapi import APIRouter

from app.api.routes import (
    auth,
    companies,
    users,
)


api_router = APIRouter()


api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"]
)


api_router.include_router(
    companies.router,
    prefix="/companies",
    tags=["Companies"]
)


api_router.include_router(
    users.router,
    prefix="/users",
    tags=["Users"]
)