from datetime import datetime

from sqlalchemy import Integer, Boolean, Unicode, text
from sqlalchemy.sql import expression

from app.db.types import UtcDateTime, utcnow
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

from typing import TYPE_CHECKING

if TYPE_CHECKING: 
    from app.models.user import User
    from app.models.customer import Customer
    from app.models.ticket import Ticket
    from app.models.call import Call

class Company(Base):
    
    __tablename__ = "companies"
    
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )
    
    name: Mapped[str] = mapped_column(
        Unicode(50),
        nullable=False
    )

    email: Mapped[str | None] = mapped_column(
        Unicode(50),
        nullable=True
    )

    phone: Mapped[str | None] = mapped_column(
        Unicode(13),
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
    
    users: Mapped[list["User"]] = relationship(
        back_populates="company"
    )
    
    customers: Mapped[list["Customer"]] = relationship(
    back_populates="company"
    )
    
    tickets: Mapped[list["Ticket"]] = relationship(
        back_populates="company"
    )

    calls: Mapped[list["Call"]] = relationship(
        back_populates="company"
    )
