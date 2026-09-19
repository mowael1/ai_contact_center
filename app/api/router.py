from fastapi import APIRouter

from app.api.routes import (
    auth,
    companies,
    customers,
    tickets,
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

api_router.include_router(
    customers.router,
    prefix="/customers",
    tags=["Customers"]
)

api_router.include_router(
    tickets.router,
    prefix="/tickets",
    tags=["Tickets"]
)