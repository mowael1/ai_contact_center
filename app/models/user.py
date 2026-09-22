from datetime import datetime

from sqlalchemy import (
    DateTime,
    Integer,
    Boolean,
    ForeignKey,
    Unicode,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

from typing import TYPE_CHECKING

if TYPE_CHECKING: 
    from app.models.company import Company
    from app.models.role import Role
    from app.models.ticket import Ticket
    from app.models.call import Call
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    company_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("companies.id"),
        nullable=True
    )

    role_id: Mapped[int] = mapped_column(
        ForeignKey("roles.id"),
        nullable=False
    )

    full_name: Mapped[str] = mapped_column(
        Unicode(50),
        nullable=False
    )

    email: Mapped[str] = mapped_column(
        Unicode(50),
        nullable=False,
        unique=True
    )

    password_hash: Mapped[str] = mapped_column(
        Unicode(255),
        nullable=False
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true")
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
    
    company: Mapped["Company | None"] = relationship(
        back_populates="users"
    )
    
    role: Mapped["Role"] = relationship(
        back_populates="users"
    )
    
    assigned_tickets: Mapped[list["Ticket"]] = relationship(
    back_populates="assigned_agent",
    foreign_keys="Ticket.assigned_agent_id"
)
    calls: Mapped[list["Call"]] = relationship(
    back_populates="agent",
    foreign_keys="Call.agent_id"
)