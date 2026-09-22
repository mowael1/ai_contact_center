from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Integer,
    Boolean,
    ForeignKey,
    Unicode,
    text,
)
from sqlalchemy.sql import expression

from app.db.types import UtcDateTime, utcnow
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from app.db.base import Base


if TYPE_CHECKING:
    from app.models.company import Company
    from app.models.ticket import Ticket


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("companies.id"),
        nullable=False,
        index=True
    )

    full_name: Mapped[str] = mapped_column(
        Unicode(200),
        nullable=False
    )

    phone: Mapped[str] = mapped_column(
        Unicode(50),
        nullable=False
    )

    email: Mapped[str | None] = mapped_column(
        Unicode(320),
        nullable=True
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=expression.true()
    )

    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime,
        nullable=False,
        server_default=utcnow()
    )

    updated_at: Mapped[datetime | None] = mapped_column(
        UtcDateTime,
        nullable=True
    )

    company: Mapped["Company"] = relationship(
        back_populates="customers"
    )
    
    tickets: Mapped[list["Ticket"]] = relationship(
        back_populates="customer"
    )