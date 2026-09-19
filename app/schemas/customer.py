from datetime import datetime
from pydantic import BaseModel, ConfigDict, EmailStr

class CustomerCreate(BaseModel):
    full_name: str
    phone: str
    email: EmailStr | None = None
    is_active: bool = True


class CustomerUpdate(BaseModel):
    full_name: str | None = None
    phone: str | None = None
    email: EmailStr | None = None


class CustomerStatusUpdate(BaseModel):
    is_active: bool


class CustomerResponse(BaseModel):
    id: int
    company_id: int
    full_name: str
    phone: str
    email: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime | None

    model_config = ConfigDict(
        from_attributes=True
    )