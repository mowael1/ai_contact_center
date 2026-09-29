from typing import Literal

from pydantic import BaseModel, Field


class AnalyticsQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = Field(None, max_length=64)
    previous_metric: Literal[
        "resolved_tickets", "unresolved_tickets", "resolved_ticket_list",
        "unresolved_ticket_list", "total_tickets", "top_issues",
        "tickets_needing_agent_call",
        "customers_with_unresolved_tickets",
        "total_calls", "resolved_calls", "unresolved_calls", "unclear_calls",
        "no_answer_calls", "failed_calls", "calls_by_outcome",
        "common_call_reasons", "average_call_duration",
    ] | None = None
    previous_time_period: Literal["all_time", "this_month", "last_30_days"] | None = None


class AnalyticsQueryResponse(BaseModel):
    query: str
    answer: str
    metric: str
    time_period: str
    data: list[dict]


class AnalyticsIntent(BaseModel):
    metric: Literal[
        "resolved_tickets",
        "unresolved_tickets",
        "resolved_ticket_list",
        "unresolved_ticket_list",
        "tickets_needing_agent_call",
        "total_tickets",
        "top_issues",
        "customers_with_unresolved_tickets",
        "total_calls",
        "resolved_calls",
        "unresolved_calls",
        "unclear_calls",
        "no_answer_calls",
        "failed_calls",
        "calls_by_outcome",
        "common_call_reasons",
        "average_call_duration",
    ]
    time_period: Literal["all_time", "this_month", "last_30_days"] = "all_time"
    limit: int = Field(default=10, ge=1, le=20)
