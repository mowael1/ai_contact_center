from datetime import datetime

from sqlalchemy import ForeignKey, Index, Integer, String, UnicodeText
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.types import UtcDateTime, utcnow


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    company_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    title: Mapped[str] = mapped_column(UnicodeText, nullable=False, default="New chat")
    summary: Mapped[str] = mapped_column(UnicodeText, nullable=False, default="")
    document_ids: Mapped[str] = mapped_column(UnicodeText, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, server_default=utcnow()
    )
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, server_default=utcnow()
    )

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )

    __table_args__ = (Index("ix_chat_sessions_company_updated", "company_id", "updated_at"),)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(UnicodeText, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, server_default=utcnow()
    )

    session: Mapped[ChatSession] = relationship(back_populates="messages")
