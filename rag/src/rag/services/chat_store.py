"""Persistent short-term chat memory for the knowledge-base assistant."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from rag.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class Memory:
    summary: str
    recent: list[dict[str, str]]


class ChatStore:
    """Use the application's database, with an in-memory fallback for standalone RAG."""

    def __init__(self) -> None:
        self._fallback: dict[tuple[int, str], dict[str, Any]] = {}
        self._database_disabled = False

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def _db(self):
        if self._database_disabled:
            return None
        try:
            from app.db.session import SessionLocal
            from app.models.chat import ChatMessage, ChatSession
            return SessionLocal, ChatMessage, ChatSession
        except Exception as exc:  # standalone RAG does not require the parent app DB
            logger.debug("Persistent chat DB unavailable: %s", exc)
            self._database_disabled = True
            return None

    @staticmethod
    def _fallback_session(company_id: int, session_id: str, title: str = "New chat") -> dict[str, Any]:
        return {
            "id": session_id, "company_id": company_id, "title": title,
            "summary": "", "document_ids": [], "created_at": ChatStore._now(),
            "updated_at": ChatStore._now(), "messages": [],
        }

    def create(self, company_id: int, title: str = "New chat") -> dict[str, Any]:
        session_id = str(uuid4())
        db = self._db()
        if db:
            SessionLocal, _, ChatSession = db
            try:
                with SessionLocal() as session:
                    item = ChatSession(id=session_id, company_id=company_id, title=title)
                    session.add(item)
                    session.commit()
                    return self._serialize_session(item)
            except Exception as exc:
                logger.warning("Could not persist chat session, using fallback: %s", exc)
                self._database_disabled = True
        item = self._fallback_session(company_id, session_id, title)
        self._fallback[(company_id, session_id)] = item
        return item

    def ensure(self, company_id: int, session_id: str, title: str = "New chat") -> dict[str, Any]:
        existing = self.get(company_id, session_id)
        if existing:
            return existing
        db = self._db()
        if db:
            SessionLocal, _, ChatSession = db
            try:
                with SessionLocal() as session:
                    item = ChatSession(id=session_id, company_id=company_id, title=title)
                    session.add(item)
                    session.commit()
                    return self._serialize_session(item)
            except Exception as exc:
                logger.warning("Could not create chat session, using fallback: %s", exc)
                self._database_disabled = True
        item = self._fallback_session(company_id, session_id, title)
        self._fallback[(company_id, session_id)] = item
        return item

    def list(self, company_id: int) -> list[dict[str, Any]]:
        db = self._db()
        if db:
            SessionLocal, _, ChatSession = db
            try:
                with SessionLocal() as session:
                    items = session.query(ChatSession).filter_by(company_id=company_id).order_by(ChatSession.updated_at.desc()).all()
                    return [self._serialize_session(item) for item in items]
            except Exception as exc:
                logger.warning("Could not list persistent chat sessions: %s", exc)
                self._database_disabled = True
        return [self._serialize_session(item) for (cid, _), item in self._fallback.items() if cid == company_id]

    def get(self, company_id: int, session_id: str) -> dict[str, Any] | None:
        db = self._db()
        if db:
            SessionLocal, _, ChatSession = db
            try:
                with SessionLocal() as session:
                    item = session.query(ChatSession).filter_by(id=session_id, company_id=company_id).first()
                    return self._serialize_session(item) if item else None
            except Exception as exc:
                logger.warning("Could not load persistent chat session: %s", exc)
                self._database_disabled = True
        return self._fallback.get((company_id, session_id))

    def messages(self, company_id: int, session_id: str) -> list[dict[str, Any]] | None:
        db = self._db()
        if db:
            SessionLocal, ChatMessage, ChatSession = db
            try:
                with SessionLocal() as session:
                    owner = session.query(ChatSession.id).filter_by(id=session_id, company_id=company_id).first()
                    if not owner:
                        return None
                    rows = session.query(ChatMessage).filter_by(session_id=session_id).order_by(ChatMessage.created_at, ChatMessage.id).all()
                    return [self._serialize_message(row) for row in rows]
            except Exception as exc:
                logger.warning("Could not load persistent chat messages: %s", exc)
                self._database_disabled = True
        item = self._fallback.get((company_id, session_id))
        return None if item is None else list(item["messages"])

    def delete(self, company_id: int, session_id: str) -> bool:
        db = self._db()
        if db:
            SessionLocal, _, ChatSession = db
            try:
                with SessionLocal() as session:
                    item = session.query(ChatSession).filter_by(id=session_id, company_id=company_id).first()
                    if not item:
                        return False
                    session.delete(item)
                    session.commit()
                    return True
            except Exception as exc:
                logger.warning("Could not delete persistent chat session: %s", exc)
                self._database_disabled = True
        return self._fallback.pop((company_id, session_id), None) is not None

    def memory(self, company_id: int, session_id: str) -> Memory:
        item = self.get(company_id, session_id)
        if not item:
            return Memory("", [])
        messages = self.messages(company_id, session_id) or []
        return Memory(item.get("summary", ""), messages[-8:])

    def append(self, company_id: int, session_id: str, question: str, answer: str, document_ids: list[str]) -> None:
        self.ensure(company_id, session_id, question[:80] or "New chat")
        db = self._db()
        if db:
            SessionLocal, ChatMessage, ChatSession = db
            try:
                with SessionLocal() as session:
                    item = session.query(ChatSession).filter_by(id=session_id, company_id=company_id).first()
                    if not item:
                        return
                    item.title = item.title if item.title != "New chat" else question[:80]
                    item.document_ids = json.dumps(sorted(set(json.loads(item.document_ids or "[]") + document_ids)))
                    session.add_all([
                        ChatMessage(session_id=session_id, role="user", content=question),
                        ChatMessage(session_id=session_id, role="assistant", content=answer),
                    ])
                    session.flush()
                    rows = session.query(ChatMessage).filter_by(session_id=session_id).order_by(ChatMessage.created_at, ChatMessage.id).all()
                    item.summary = self._summary([self._serialize_message(row) for row in rows[:-8]])
                    item.updated_at = self._now()
                    session.commit()
                    return
            except Exception as exc:
                logger.warning("Could not persist chat messages, using fallback: %s", exc)
                self._database_disabled = True
                self._fallback.setdefault(
                    (company_id, session_id),
                    self._fallback_session(company_id, session_id, question[:80] or "New chat"),
                )
        item = self._fallback[(company_id, session_id)]
        item["title"] = item["title"] if item["title"] != "New chat" else question[:80]
        item["document_ids"] = sorted(set(item["document_ids"] + document_ids))
        next_id = len(item["messages"]) + 1
        item["messages"].extend([
            {"id": next_id, "session_id": session_id, "role": "user", "content": question, "created_at": self._now()},
            {"id": next_id + 1, "session_id": session_id, "role": "assistant", "content": answer, "created_at": self._now()},
        ])
        item["summary"] = self._summary(item["messages"][:-8])
        item["updated_at"] = self._now()

    @staticmethod
    def _summary(messages: list[dict[str, Any]]) -> str:
        if not messages:
            return ""
        # Extractive memory preserves names, numbers, dates and decisions without another LLM call.
        return "\n".join(f"{m['role'].title()}: {m['content']}" for m in messages)[-6000:]

    @staticmethod
    def _serialize_session(item) -> dict[str, Any]:
        if isinstance(item, dict):
            return item
        return {
            "id": item.id, "company_id": item.company_id, "title": item.title,
            "summary": item.summary, "document_ids": json.loads(item.document_ids or "[]"),
            "created_at": item.created_at, "updated_at": item.updated_at,
        }

    @staticmethod
    def _serialize_message(item) -> dict[str, Any]:
        return {"id": item.id, "session_id": item.session_id, "role": item.role,
                "content": item.content, "created_at": item.created_at}


chat_store = ChatStore()
