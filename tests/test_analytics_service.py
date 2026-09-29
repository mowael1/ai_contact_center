from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.models import Call, Ticket
from app.schemas.analytics import AnalyticsIntent
from app.services.analytics_service import _follow_up_list_intent, _run_query
from app.services import analytics_service


@pytest.fixture
def analytics_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    now = datetime.now(timezone.utc)
    session.add_all([
        Ticket(
            id=1, company_id=1, customer_id=10, assigned_agent_id=100,
            subject="Internet", status="resolved", created_at=datetime.now(timezone.utc),
        ),
        Ticket(
            id=2, company_id=1, customer_id=11, assigned_agent_id=100,
            subject="Internet", status="needs_agent", created_at=datetime.now(timezone.utc),
        ),
        Ticket(
            id=3, company_id=1, customer_id=11, assigned_agent_id=100,
            subject="Billing", status="needs_agent", created_at=datetime.now(timezone.utc),
        ),
        Ticket(
            id=5, company_id=1, customer_id=12, assigned_agent_id=100,
            subject="New issue", status="pending_follow_up", created_at=datetime.now(timezone.utc),
        ),
        Ticket(
            id=4, company_id=2, customer_id=20, assigned_agent_id=200,
            subject="Other company issue", status="resolved", created_at=now,
        ),
        Call(
            id=1, company_id=1, ticket_id=1, customer_id=10, agent_id=100,
            status="completed", outcome="resolved", duration_seconds=60,
            reason="Internet connection", created_at=now,
        ),
        Call(
            id=2, company_id=1, ticket_id=2, customer_id=11, agent_id=100,
            status="completed", outcome="not_resolved", duration_seconds=180,
            reason="Internet connection", created_at=now,
        ),
        Call(
            id=3, company_id=1, ticket_id=3, customer_id=11, agent_id=100,
            status="no_answer", outcome=None, duration_seconds=None,
            reason=None, created_at=now,
        ),
        Call(
            id=4, company_id=2, ticket_id=4, customer_id=20, agent_id=200,
            status="completed", outcome="resolved", duration_seconds=90,
            reason="Billing", created_at=now,
        ),
    ])
    session.commit()
    yield session
    session.close()
    engine.dispose()


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("resolved_tickets", [{"resolved_tickets": 1}]),
        ("unresolved_tickets", [{"unresolved_tickets": 3}]),
        ("total_tickets", [{"total_tickets": 4}]),
        ("customers_with_unresolved_tickets", [{"customers_with_unresolved_tickets": 2}]),
    ],
)
def test_ticket_counts_are_company_scoped(analytics_db, metric, expected):
    result = _run_query(analytics_db, 1, AnalyticsIntent(metric=metric))
    assert result == expected


def test_top_issues_returns_subjects_ranked_by_ticket_count(analytics_db):
    intent = AnalyticsIntent(metric="top_issues", limit=5)
    assert _run_query(analytics_db, 1, intent) == [
        {"subject": "Internet", "ticket_count": 2},
        {"subject": "Billing", "ticket_count": 1},
        {"subject": "New issue", "ticket_count": 1},
    ]


def test_time_period_limits_ticket_counts(analytics_db):
    analytics_db.add(Ticket(
        id=6, company_id=1, customer_id=13, assigned_agent_id=100,
        subject="Older issue", status="resolved",
        created_at=datetime.now(timezone.utc) - timedelta(days=60),
    ))
    analytics_db.commit()

    assert _run_query(
        analytics_db, 1, AnalyticsIntent(metric="resolved_tickets", time_period="last_30_days")
    ) == [{"resolved_tickets": 1}]
    assert _run_query(
        analytics_db, 1, AnalyticsIntent(metric="resolved_tickets", time_period="all_time")
    ) == [{"resolved_tickets": 2}]


def test_analytics_sql_is_written_to_application_terminal_log(analytics_db, caplog):
    with caplog.at_level("INFO", logger="uvicorn.error"):
        _run_query(analytics_db, 1, AnalyticsIntent(metric="resolved_tickets"))

    assert "Analytics SQL:" in caplog.text
    assert "tickets.company_id" in caplog.text
    assert "SQL parameters:" in caplog.text


def test_analytics_intent_rejects_unsupported_metric():
    with pytest.raises(ValueError):
        AnalyticsIntent(metric="drop_table")


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("total_calls", [{"total_calls": 3}]),
        ("resolved_calls", [{"resolved_calls": 1}]),
        ("unresolved_calls", [{"unresolved_calls": 1}]),
        ("no_answer_calls", [{"no_answer_calls": 1}]),
        ("failed_calls", [{"failed_calls": 0}]),
    ],
)
def test_call_counts_are_company_scoped(analytics_db, metric, expected):
    assert _run_query(analytics_db, 1, AnalyticsIntent(metric=metric)) == expected


def test_call_analytics_summarize_outcomes_reasons_and_duration(analytics_db):
    assert _run_query(analytics_db, 1, AnalyticsIntent(metric="common_call_reasons")) == [
        {"reason": "Internet connection", "call_count": 2},
        {"reason": "Billing", "call_count": 1},
    ]
    assert _run_query(analytics_db, 1, AnalyticsIntent(metric="average_call_duration")) == [
        {"average_duration_seconds": 120.0},
    ]
    assert _run_query(analytics_db, 1, AnalyticsIntent(metric="calls_by_outcome")) == [
        {"outcome": "no_answer", "call_count": 1},
        {"outcome": "not_resolved", "call_count": 1},
        {"outcome": "resolved", "call_count": 1},
    ]


def test_egyptian_follow_up_keeps_previous_status_and_period():
    intent = _follow_up_list_intent(
        "إيه هما؟", "unresolved_tickets", "last_30_days"
    )
    assert intent is not None
    assert intent.metric == "unresolved_ticket_list"
    assert intent.time_period == "last_30_days"


def test_unresolved_follow_up_lists_only_unresolved_company_tickets(analytics_db):
    intent = AnalyticsIntent(metric="unresolved_ticket_list")
    assert _run_query(analytics_db, 1, intent) == [
        {"subject": "Billing", "status": "needs_agent", "ticket_count": 1},
        {"subject": "Internet", "status": "needs_agent", "ticket_count": 1},
        {"subject": "New issue", "status": "pending_follow_up", "ticket_count": 1},
    ]


def test_agent_call_list_uses_pending_follow_up_status_and_returns_total(analytics_db):
    assert _run_query(
        analytics_db, 1, AnalyticsIntent(metric="tickets_needing_agent_call")
    ) == [{
        "total": 1,
        "returned": 1,
        "tickets": [{"ticket_id": 5, "subject": "New issue"}],
    }]


def test_follow_up_asking_if_the_list_is_complete_checks_agent_call_list():
    intent = _follow_up_list_intent(
        "دول كلهم ولا لا؟", "unresolved_ticket_list", "all_time"
    )
    assert intent is not None
    assert intent.metric == "tickets_needing_agent_call"


def test_conversation_history_is_passed_to_intent_parser(analytics_db, monkeypatch):
    captured = {}

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    message = SimpleNamespace(content="Five tickets remain unresolved.")
                    return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def parse_intent(client, question, history=""):
        captured["history"] = history
        return AnalyticsIntent(metric="unresolved_tickets")

    monkeypatch.setattr(analytics_service, "_client", lambda: FakeClient())
    monkeypatch.setattr(analytics_service, "_parse_intent", parse_intent)
    user = SimpleNamespace(company_id=1)

    result = analytics_service.answer_analytics_question(
        analytics_db,
        user,
        "طب مين فيهم لسه؟",
        history="User: كام مشكلة متحلتش؟\nAssistant: في 3 تذاكر.",
    )

    assert "كام مشكلة متحلتش" in captured["history"]
    assert result["data"] == [{"unresolved_tickets": 3}]
