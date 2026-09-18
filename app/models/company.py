from datetime import datetime

from sqlalchemy import Integer, Boolean, Unicode, text
from sqlalchemy.dialects.mssql import DATETIME2
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

from typing import TYPE_CHECKING

if TYPE_CHECKING: 
    from app.models.user import User

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
        server_default=text("1")
    )

    created_at: Mapped[datetime] = mapped_column(
        DATETIME2,
        nullable=False,
        server_default=text("SYSUTCDATETIME()")
    )

    updated_at: Mapped[datetime | None] = mapped_column(
        DATETIME2,
        nullable=True
    )
    
    users: Mapped[list["User"]] = relationship(
        back_populates="company"
    )