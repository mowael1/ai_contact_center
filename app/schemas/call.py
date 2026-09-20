from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class CallResponse(BaseModel):
    id: int

    company_id: int
    ticket_id: int
    customer_id: int
    agent_id: int

    provider_call_id: str | None

    status: str
    outcome: str | None

    transcript: str | None

    started_at: datetime | None
    ended_at: datetime | None
    duration_seconds: int | None

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )


class MockCallResult(BaseModel):
    result: Literal[
        "resolved",
        "not_resolved",
        "unclear",
        "no_answer",
        "failed",
    ]