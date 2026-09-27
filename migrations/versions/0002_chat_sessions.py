"""Add persistent knowledge-base chat sessions and messages.

Revision ID: 0002_chat_sessions
Revises: 6f6f4d6d5cee
"""

from alembic import op
import sqlalchemy as sa

from app.db.types import UtcDateTime, utcnow

revision = "0002_chat_sessions"
down_revision = "6f6f4d6d5cee"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.UnicodeText(), nullable=False, server_default="New chat"),
        sa.Column("summary", sa.UnicodeText(), nullable=False, server_default=""),
        sa.Column("document_ids", sa.UnicodeText(), nullable=False, server_default="[]"),
        sa.Column("created_at", UtcDateTime(), nullable=False, server_default=utcnow()),
        sa.Column("updated_at", UtcDateTime(), nullable=False, server_default=utcnow()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chat_sessions_company_id", "chat_sessions", ["company_id"])
    op.create_index(
        "ix_chat_sessions_company_updated", "chat_sessions", ["company_id", "updated_at"]
    )

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.UnicodeText(), nullable=False),
        sa.Column("created_at", UtcDateTime(), nullable=False, server_default=utcnow()),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_chat_messages_session_id", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_index("ix_chat_sessions_company_updated", table_name="chat_sessions")
    op.drop_index("ix_chat_sessions_company_id", table_name="chat_sessions")
    op.drop_table("chat_sessions")
