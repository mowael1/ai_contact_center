from datetime import datetime

from pydantic import BaseModel, ConfigDict


class TicketCreate(BaseModel):
    customer_id: int
    assigned_agent_id: int

    subject: str
    description: str | None = None
    procedure_steps: str | None = None


class TicketResponse(BaseModel):
    id: int
    company_id: int
    customer_id: int
    assigned_agent_id: int

    subject: str
    description: str | None
    procedure_steps: str | None

    status: str

    created_at: datetime
    updated_at: datetime | None

    model_config = ConfigDict(
        from_attributes=True
    )
    
class TicketCustomerBrief(BaseModel):
    id: int
    full_name: str
    phone: str
    email: str | None

    model_config = ConfigDict(
        from_attributes=True
    )


class FollowUpTicketResponse(BaseModel):
    id: int
    subject: str
    description: str | None
    procedure_steps: str | None
    status: str

    customer: TicketCustomerBrief

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )