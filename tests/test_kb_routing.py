"""Regression tests for the Knowledge Base integration routes."""

from app.main import app


def test_knowledge_base_routes_are_registered() -> None:
    paths = app.openapi()["paths"]

    assert "/api/v1/kb/documents" in paths
    assert "/api/v1/kb/sessions" in paths
