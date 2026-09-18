from app.db.base import Base

from sqlalchemy import Integer, Unicode
from sqlalchemy.orm import mapped_column, Mapped, relationship

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.user import User


class Role(Base):
    
    __tablename__ = "roles"
    
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )
    
    name: Mapped[str] = mapped_column(
        Unicode(50),
        nullable=False,
        unique=True
    )
    
    users: Mapped[list["User"]] = relationship(
        back_populates="role"
    )