"""How a request's company is resolved - the control that keeps tenants apart.

The bug these cover: in dev mode a missing ``X-Company-Id`` header used to fall
back to a default company, so every admin resolved to the same tenant and saw
one company's documents. A missing scope must fail, never default.
"""

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from rag.api.deps import get_tenant
from rag.config import settings
from rag.services.tenancy import Tenant


@pytest.fixture
def client():
    app = FastAPI()

    @app.get("/whoami")
    def whoami(tenant: Tenant = Depends(get_tenant)):
        return {"company_id": tenant.company_id, "collection": tenant.collection_name}

    return TestClient(app, raise_server_exceptions=False)


# ---- dev mode -------------------------------------------------------------
def test_dev_mode_requires_an_explicit_company(client, monkeypatch):
    """No header must be an error, not a default company."""
    monkeypatch.setattr(settings, "RAG_AUTH_MODE", "dev")
    response = client.get("/whoami")
    assert response.status_code == 400
    assert "X-Company-Id" in response.json()["detail"]


def test_dev_mode_uses_the_header(client, monkeypatch):
    monkeypatch.setattr(settings, "RAG_AUTH_MODE", "dev")
    body = client.get("/whoami", headers={"X-Company-Id": "42"}).json()
    assert body["company_id"] == 42
    assert body["collection"] == "kb_company_42"


def test_dev_mode_rejects_a_non_positive_company(client, monkeypatch):
    monkeypatch.setattr(settings, "RAG_AUTH_MODE", "dev")
    assert client.get("/whoami", headers={"X-Company-Id": "0"}).status_code == 400
    assert client.get("/whoami", headers={"X-Company-Id": "-3"}).status_code == 400


def test_two_companies_resolve_to_different_collections(client, monkeypatch):
    monkeypatch.setattr(settings, "RAG_AUTH_MODE", "dev")
    a = client.get("/whoami", headers={"X-Company-Id": "1"}).json()
    b = client.get("/whoami", headers={"X-Company-Id": "2"}).json()
    assert a["collection"] != b["collection"]


# ---- jwt mode -------------------------------------------------------------
class FakeUser:
    def __init__(self, company_id):
        self.company_id = company_id


@pytest.fixture
def jwt_client(monkeypatch):
    """A client whose authenticated user is controllable per test.

    The user is swapped with FastAPI's dependency_overrides rather than by
    patching the module attribute: ``Depends`` captures the function object at
    import time, so patching the name would have no effect.
    """
    monkeypatch.setattr(settings, "RAG_AUTH_MODE", "jwt")
    holder = {"user": None}

    import rag.api.deps as deps

    app = FastAPI()

    @app.get("/whoami")
    def whoami(tenant: Tenant = Depends(deps.get_tenant)):
        return {"company_id": tenant.company_id}

    app.dependency_overrides[deps._current_user] = lambda: holder["user"]
    return TestClient(app, raise_server_exceptions=False), holder


def test_jwt_mode_takes_the_company_from_the_user(jwt_client):
    client, holder = jwt_client
    holder["user"] = FakeUser(company_id=7)
    assert client.get("/whoami").json()["company_id"] == 7


def test_jwt_mode_ignores_a_client_supplied_company(jwt_client):
    """A header must never override the signed-in user's company."""
    client, holder = jwt_client
    holder["user"] = FakeUser(company_id=7)
    body = client.get("/whoami", headers={"X-Company-Id": "9999"}).json()
    assert body["company_id"] == 7


def test_jwt_mode_refuses_a_user_without_a_company(jwt_client):
    """A super admin has no company, so has no knowledge base - not all of them."""
    client, holder = jwt_client
    holder["user"] = FakeUser(company_id=None)
    response = client.get("/whoami")
    assert response.status_code == 403
    assert "not attached to a company" in response.json()["detail"]


def test_jwt_mode_refuses_an_unauthenticated_request(jwt_client):
    """No user resolved means no company, so the request is refused."""
    client, holder = jwt_client
    holder["user"] = None
    assert client.get("/whoami").status_code == 403


# ---- super admin delegation ----------------------------------------------
class FakeRole:
    def __init__(self, name):
        self.name = name


class FakeUserWithRole:
    def __init__(self, uid, company_id, role):
        self.id, self.company_id, self.role = uid, company_id, FakeRole(role)


def test_super_admin_must_choose_a_company(jwt_client):
    """There is no "see everything" scope; a company has to be named."""
    client, holder = jwt_client
    holder["user"] = FakeUserWithRole(1, None, "super_admin")
    response = client.get("/whoami")
    assert response.status_code == 400
    assert "Choose a company" in response.json()["detail"]


def test_super_admin_can_act_on_a_chosen_company(jwt_client):
    client, holder = jwt_client
    holder["user"] = FakeUserWithRole(1, None, "super_admin")
    assert client.get("/whoami", headers={"X-Company-Id": "202"}).json()["company_id"] == 202


def test_super_admin_role_check_is_case_insensitive(jwt_client):
    client, holder = jwt_client
    holder["user"] = FakeUserWithRole(1, None, "SUPER_ADMIN")
    assert client.get("/whoami", headers={"X-Company-Id": "5"}).status_code == 200


def test_a_companyless_non_super_admin_cannot_delegate(jwt_client):
    """Delegation is granted by role, not merely by having no company."""
    client, holder = jwt_client
    holder["user"] = FakeUserWithRole(2, None, "agent")
    response = client.get("/whoami", headers={"X-Company-Id": "202"})
    assert response.status_code == 403


def test_an_admin_cannot_borrow_another_company(jwt_client):
    """A user who has a company always gets their own, header or not."""
    client, holder = jwt_client
    holder["user"] = FakeUserWithRole(3, 101, "admin")
    body = client.get("/whoami", headers={"X-Company-Id": "202"}).json()
    assert body["company_id"] == 101


def test_super_admin_rejects_an_invalid_company(jwt_client):
    client, holder = jwt_client
    holder["user"] = FakeUserWithRole(1, None, "super_admin")
    assert client.get("/whoami", headers={"X-Company-Id": "0"}).status_code == 400


def test_delegation_is_flagged_on_the_tenant(monkeypatch):
    """The borrowed scope must be visible, not silent."""
    import rag.api.deps as deps

    monkeypatch.setattr(settings, "RAG_AUTH_MODE", "jwt")
    tenant = deps.get_tenant(202, FakeUserWithRole(1, None, "super_admin"))
    assert tenant.company_id == 202
    assert tenant.is_delegated is True


def test_own_company_is_not_flagged_as_delegated(monkeypatch):
    import rag.api.deps as deps

    monkeypatch.setattr(settings, "RAG_AUTH_MODE", "jwt")
    user = FakeUserWithRole(3, 101, "admin")
    tenant = deps.get_tenant(None, user)
    assert tenant.company_id == 101
    assert tenant.is_delegated is False


# ---- detached ORM instances ----------------------------------------------
def test_role_is_read_without_touching_a_closed_session(monkeypatch):
    """Regression: resolving the role must not lazy-load after session close.

    ``_current_user`` used to return the ORM object, so the first ``user.role``
    access raised DetachedInstanceError once the session had closed.
    """
    import rag.api.deps as deps

    class DetachedRole:
        @property
        def name(self):
            raise RuntimeError("lazy load on a detached instance")

    class DetachedUser:
        id = 1
        company_id = None
        role = DetachedRole()

    monkeypatch.setattr(settings, "RAG_AUTH_MODE", "jwt")

    # The identity carries plain values, so nothing is lazy-loaded.
    identity = deps.CallerIdentity(user_id=1, company_id=None, role="super_admin")
    tenant = deps.get_tenant(77, identity)
    assert tenant.company_id == 77 and tenant.is_delegated

    # And the raw detached object would indeed have blown up.
    with pytest.raises(RuntimeError):
        _ = DetachedUser().role.name


def test_caller_identity_carries_only_plain_values():
    import rag.api.deps as deps

    identity = deps.CallerIdentity(user_id=3, company_id=9, role="admin")
    assert isinstance(identity.role, str)
    assert deps._role_name(identity) == "admin"
