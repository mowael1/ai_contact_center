from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.schemas.ticket import TicketUpdate
from app.services import ticket_service


def test_frontend_serves_current_ticket_edit_ui_without_cache() -> None:
    client = TestClient(app)
    redirect = client.get("/", follow_redirects=False)

    assert redirect.status_code == 307
    assert "no-store" in redirect.headers["cache-control"]
    assert redirect.headers["clear-site-data"] == '"cache"'
    assert redirect.headers["location"].startswith("/static/index.html?v=")

    response = client.get(redirect.headers["location"])

    assert response.status_code == 200
    assert "renderTicketEdit" in response.text
    assert "ticketEditStatus" in response.text
    assert "window.history.replaceState" in response.text


def test_ticket_patch_route_is_registered() -> None:
    operation = app.openapi()["paths"]["/api/v1/tickets/{ticket_id}"]

    assert "patch" in operation


def test_ticket_update_rejects_protected_fields() -> None:
    with pytest.raises(ValidationError):
        TicketUpdate.model_validate({"company_id": 8})


def test_ticket_update_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        TicketUpdate.model_validate({"status": "closed"})


def test_admin_can_update_company_ticket(monkeypatch) -> None:
    db = object()
    admin = SimpleNamespace(company_id=7)
    ticket = SimpleNamespace(id=12, company_id=7)
    customer = SimpleNamespace(id=9, company_id=7, is_active=True)
    agent = SimpleNamespace(id=4, company_id=7, is_active=True)
    captured = {}

    monkeypatch.setattr(
        ticket_service.ticket_repository,
        "get_by_id_and_company",
        lambda session, ticket_id, company_id: ticket,
    )
    monkeypatch.setattr(
        ticket_service.customer_repository,
        "get_by_id_and_company",
        lambda session, customer_id, company_id: customer,
    )
    monkeypatch.setattr(
        ticket_service.user_repository,
        "get_agent_by_id_and_company",
        lambda session, agent_id, company_id: agent,
    )

    def fake_update(session, existing_ticket, data):
        captured["db"] = session
        captured["ticket"] = existing_ticket
        captured["data"] = data.model_dump(exclude_unset=True)
        return existing_ticket

    monkeypatch.setattr(
        ticket_service.ticket_repository,
        "update",
        fake_update,
    )

    result = ticket_service.update_ticket(
        db,
        admin,
        ticket.id,
        TicketUpdate(
            customer_id=customer.id,
            assigned_agent_id=agent.id,
            subject="  Updated issue  ",
            description="New description",
            status="resolved",
        ),
    )

    assert result is ticket
    assert captured == {
        "db": db,
        "ticket": ticket,
        "data": {
            "customer_id": 9,
            "assigned_agent_id": 4,
            "subject": "Updated issue",
            "description": "New description",
            "status": "resolved",
        },
    }


def test_admin_cannot_update_ticket_from_another_company(monkeypatch) -> None:
    monkeypatch.setattr(
        ticket_service.ticket_repository,
        "get_by_id_and_company",
        lambda session, ticket_id, company_id: None,
    )

    with pytest.raises(HTTPException) as exc_info:
        ticket_service.update_ticket(
            object(),
            SimpleNamespace(company_id=7),
            99,
            TicketUpdate(subject="Updated issue"),
        )

    assert exc_info.value.status_code == 404


def test_admin_cannot_assign_inactive_agent(monkeypatch) -> None:
    monkeypatch.setattr(
        ticket_service.ticket_repository,
        "get_by_id_and_company",
        lambda session, ticket_id, company_id: SimpleNamespace(id=ticket_id),
    )
    monkeypatch.setattr(
        ticket_service.user_repository,
        "get_agent_by_id_and_company",
        lambda session, agent_id, company_id: SimpleNamespace(is_active=False),
    )

    with pytest.raises(HTTPException) as exc_info:
        ticket_service.update_ticket(
            object(),
            SimpleNamespace(company_id=7),
            12,
            TicketUpdate(assigned_agent_id=4),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Agent is inactive"


def test_assigned_agent_cannot_be_cleared(monkeypatch) -> None:
    monkeypatch.setattr(
        ticket_service.ticket_repository,
        "get_by_id_and_company",
        lambda session, ticket_id, company_id: SimpleNamespace(id=ticket_id),
    )

    with pytest.raises(HTTPException) as exc_info:
        ticket_service.update_ticket(
            object(),
            SimpleNamespace(company_id=7),
            12,
            TicketUpdate(assigned_agent_id=None),
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "Assigned agent is required"


def test_admin_cannot_assign_inactive_customer(monkeypatch) -> None:
    monkeypatch.setattr(
        ticket_service.ticket_repository,
        "get_by_id_and_company",
        lambda session, ticket_id, company_id: SimpleNamespace(id=ticket_id),
    )
    monkeypatch.setattr(
        ticket_service.customer_repository,
        "get_by_id_and_company",
        lambda session, customer_id, company_id: SimpleNamespace(is_active=False),
    )

    with pytest.raises(HTTPException) as exc_info:
        ticket_service.update_ticket(
            object(),
            SimpleNamespace(company_id=7),
            12,
            TicketUpdate(customer_id=9),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Customer is inactive"


def test_ticket_subject_cannot_be_blank(monkeypatch) -> None:
    monkeypatch.setattr(
        ticket_service.ticket_repository,
        "get_by_id_and_company",
        lambda session, ticket_id, company_id: SimpleNamespace(id=ticket_id),
    )

    with pytest.raises(HTTPException) as exc_info:
        ticket_service.update_ticket(
            object(),
            SimpleNamespace(company_id=7),
            12,
            TicketUpdate(subject="   "),
        )

    assert exc_info.value.status_code == 422
