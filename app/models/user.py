from datetime import datetime

from sqlalchemy import (
    Integer,
    Boolean,
    ForeignKey,
    Unicode,
    text,
)
from sqlalchemy.dialects.mssql import DATETIME2
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

from typing import TYPE_CHECKING

if TYPE_CHECKING: 
    from app.models.company import Company
    from app.models.role import Role

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
    
    company: Mapped["Company | None"] = relationship(
        back_populates="users"
    )
    
    role: Mapped["Role"] = relationship(
        back_populates="users"
    )