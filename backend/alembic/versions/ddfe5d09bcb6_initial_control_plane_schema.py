"""initial_control_plane_schema

Revision ID: ddfe5d09bcb6
Revises:
Create Date: 2026-02-24 21:16:33.746348

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "ddfe5d09bcb6"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all 9 control-plane tables."""

    # 1. workspace
    op.create_table(
        "workspace",
        sa.Column("workspace_id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(255), unique=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # 2. workspace_member
    op.create_table(
        "workspace_member",
        sa.Column("user_id", sa.String(255), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.Uuid(),
            sa.ForeignKey("workspace.workspace_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("joined_at", sa.DateTime(timezone=True), nullable=True),
    )

    # 3. experiment_job
    op.create_table(
        "experiment_job",
        sa.Column("job_id", sa.Uuid(), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.Uuid(),
            sa.ForeignKey("workspace.workspace_id"),
            nullable=False,
        ),
        sa.Column("spec_ref", sa.String(512), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("progress_json", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(255), nullable=False),
    )

    # 4. job_event
    op.create_table(
        "job_event",
        sa.Column(
            "job_id",
            sa.Uuid(),
            sa.ForeignKey("experiment_job.job_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("sequence", sa.Integer(), primary_key=True),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )

    # 5. modeling_session
    op.create_table(
        "modeling_session",
        sa.Column("session_id", sa.Uuid(), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.Uuid(),
            sa.ForeignKey("workspace.workspace_id"),
            nullable=False,
        ),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("selected_silver_id", sa.String(512), nullable=True),
        sa.Column("model_id", sa.String(128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(255), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # 6. modeling_step_state
    op.create_table(
        "modeling_step_state",
        sa.Column(
            "session_id",
            sa.Uuid(),
            sa.ForeignKey("modeling_session.session_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("step_name", sa.String(64), primary_key=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=True),
        sa.Column("committed_at", sa.DateTime(timezone=True), nullable=True),
    )

    # 7. ingestion_live_session
    op.create_table(
        "ingestion_live_session",
        sa.Column("session_id", sa.Uuid(), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.Uuid(),
            sa.ForeignKey("workspace.workspace_id"),
            nullable=False,
        ),
        sa.Column("symbol", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("checkpoint", sa.JSON(), nullable=True),
        sa.Column("config_json", sa.JSON(), nullable=True),
    )

    # 8. serving_activation
    op.create_table(
        "serving_activation",
        sa.Column("activation_id", sa.Uuid(), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.Uuid(),
            sa.ForeignKey("workspace.workspace_id"),
            nullable=False,
        ),
        sa.Column("alias", sa.String(128), nullable=False),
        sa.Column("serving_id", sa.String(512), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("activated_by", sa.String(255), nullable=False),
        sa.Column("deactivated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
    )

    # 9. audit_event
    op.create_table(
        "audit_event",
        sa.Column("event_id", sa.Uuid(), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.Uuid(),
            sa.ForeignKey("workspace.workspace_id"),
            nullable=True,
        ),
        sa.Column("actor_id", sa.String(255), nullable=False),
        sa.Column("action", sa.String(128), nullable=False),
        sa.Column("target", sa.String(512), nullable=True),
        sa.Column("payload_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_audit_event_created_at", "audit_event", ["created_at"])
    op.create_index("ix_audit_event_workspace_id", "audit_event", ["workspace_id"])


def downgrade() -> None:
    """Drop all 9 control-plane tables in reverse dependency order."""
    op.drop_index("ix_audit_event_workspace_id", table_name="audit_event")
    op.drop_index("ix_audit_event_created_at", table_name="audit_event")
    op.drop_table("audit_event")
    op.drop_table("serving_activation")
    op.drop_table("ingestion_live_session")
    op.drop_table("modeling_step_state")
    op.drop_table("modeling_session")
    op.drop_table("job_event")
    op.drop_table("experiment_job")
    op.drop_table("workspace_member")
    op.drop_table("workspace")
