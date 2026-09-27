"""Add the reason captured for unresolved calls.

Revision ID: 6f6f4d6d5cee
Revises: 0001_initial
"""

from alembic import op
import sqlalchemy as sa

revision = "6f6f4d6d5cee"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("calls", sa.Column("reason", sa.UnicodeText(), nullable=True))


def downgrade() -> None:
    op.drop_column("calls", "reason")
