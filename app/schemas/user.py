from datetime import datetime
from pydantic import BaseModel, ConfigDict, EmailStr


class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    company_id: int | None = None
    role_id: int
    is_active: bool = True


class RoleBrief(BaseModel):
    id: int
    name: str

    model_config = ConfigDict(
        from_attributes=True
    )


class CompanyBrief(BaseModel):
    id: int
    name: str

    model_config = ConfigDict(
        from_attributes=True
    )


class UserResponse(BaseModel):
    id: int
    company_id: int | None
    role_id: int
    full_name: str
    email: str
    is_active: bool
    created_at: datetime
    updated_at: datetime | None

    model_config = ConfigDict(
        from_attributes=True
    )


class CurrentUserResponse(BaseModel):
    id: int
    full_name: str
    email: str
    is_active: bool

    role: RoleBrief
    company: CompanyBrief | None

    model_config = ConfigDict(
        from_attributes=True
    )
    
# Super Admin will send only these values
class CompanyAdminCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    is_active: bool = True
    
    
class AgentCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    is_active: bool = True
    

class AgentUpdate(BaseModel):
    full_name: str | None = None
    email: EmailStr | None = None


class UserStatusUpdate(BaseModel):
    is_active: bool
    
class AdminUpdate(BaseModel):
    full_name: str | None = None
    email: EmailStr | None = None
    
class UserStatusUpdate(BaseModel):
    is_active: bool