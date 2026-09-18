from datetime import datetime

from pydantic import BaseModel, ConfigDict

class CompanyCreate(BaseModel):
    name: str
    email: str | None = None
    phone: str | None = None
    is_active: bool = True
    
class CompanyUpdate(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None


class CompanyStatusUpdate(BaseModel):
    is_active: bool
    

class CompanyResponse(BaseModel):
    id: int
    name: str
    email: str | None
    phone: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime | None

    model_config = ConfigDict(
        from_attributes=True
    )
    
class CompanyStatusUpdate(BaseModel):
    is_active: bool