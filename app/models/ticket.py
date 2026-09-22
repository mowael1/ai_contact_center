from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    Integer,
    ForeignKey,
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
    from app.models.user import User
    from app.models.call import Call

class Ticket(Base):
    __tablename__ = "tickets"

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

    customer_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("customers.id"),
        nullable=False
    )

    assigned_agent_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id"),
        nullable=False
    )

    subject: Mapped[str] = mapped_column(
        Unicode(200),
        nullable=False
    )

    description: Mapped[str | None] = mapped_column(
        UnicodeText,
        nullable=True
    )

    procedure_steps: Mapped[str | None] = mapped_column(
        UnicodeText,
        nullable=True
    )

    status: Mapped[str] = mapped_column(
        Unicode(50),
        nullable=False,
        server_default=text("'pending_follow_up'")
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("(now() at time zone 'utc')")
    )

    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True
    )

    company: Mapped["Company"] = relationship(
        back_populates="tickets"
    )

    customer: Mapped["Customer"] = relationship(
        back_populates="tickets"
    )

    assigned_agent: Mapped["User"] = relationship(
        back_populates="assigned_tickets",
        foreign_keys=[assigned_agent_id]
    )
    
    calls: Mapped[list["Call"]] = relationship(
    back_populates="ticket"
)