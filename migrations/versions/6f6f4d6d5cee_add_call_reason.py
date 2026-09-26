"""add call reason

Revision ID: 6f6f4d6d5cee
Revises: 0001_initial
Create Date: 2026-09-23 20:40:09.133776

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6f6f4d6d5cee'
down_revision: Union[str, Sequence[str], None] = '0001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'calls',
        sa.Column('reason', sa.UnicodeText(), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('calls', 'reason')