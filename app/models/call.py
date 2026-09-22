from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    Unicode,
    UnicodeText,
    text,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from app.db.base import Base


if TYPE_CHECKING:
    from app.models.company import Company
    from app.models.customer import Customer
    from app.models.ticket import Ticket
    from app.models.user import User


class Call(Base):
    __tablename__ = "calls"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("companies.id"),
        nullable=False
    )

    ticket_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("tickets.id"),
        nullable=False
    )

    customer_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("customers.id"),
        nullable=False
    )

    agent_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id"),
        nullable=False
    )

    provider_call_id: Mapped[str | None] = mapped_column(
        Unicode(100),
        nullable=True
    )

    status: Mapped[str] = mapped_column(
        Unicode(50),
        nullable=False,
        server_default=text("'initiating'")
    )

    outcome: Mapped[str | None] = mapped_column(
        Unicode(50),
        nullable=True
    )

    transcript: Mapped[str | None] = mapped_column(
        UnicodeText,
        nullable=True
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True
    )

    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True
    )

    duration_seconds: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("(now() at time zone 'utc')")
    )

    company: Mapped["Company"] = relationship(
        back_populates="calls"
    )

    ticket: Mapped["Ticket"] = relationship(
        back_populates="calls"
    )

    customer: Mapped["Customer"] = relationship(
        back_populates="calls"
    )

    agent: Mapped["User"] = relationship(
        back_populates="calls",
        foreign_keys=[agent_id]
    )