from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = '0001_init'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'market_data',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('symbol', sa.String(16), nullable=False),
        sa.Column('timeframe', sa.String(8), nullable=False),
        sa.Column('ts_utc', sa.BigInteger, nullable=False),
        sa.Column('open_t', sa.Float, nullable=False),
        sa.Column('high_t', sa.Float, nullable=False),
        sa.Column('low_t', sa.Float, nullable=False),
        sa.Column('close_t', sa.Float, nullable=False),
        sa.Column('volume_t', sa.Float, nullable=True),
        sa.Column('resampled', sa.Boolean, nullable=False, server_default=sa.text('0')),
        sa.Column('ingested_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.UniqueConstraint('symbol', 'timeframe', 'ts_utc', name='uq_candle_key'),
        sa.Index('ix_md_symbol_tf_ts', 'symbol', 'timeframe', 'ts_utc')
    )

    op.create_table(
        'features',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('symbol', sa.String(16), nullable=False),
        sa.Column('timeframe', sa.String(8), nullable=False),
        sa.Column('ts_utc', sa.BigInteger, nullable=False),
        sa.Column('version', sa.String(32), nullable=False),
        sa.Column('data', sa.LargeBinary, nullable=False),  # serialized via pyarrow
        sa.UniqueConstraint('symbol', 'timeframe', 'ts_utc', 'version', name='uq_feature_key'),
        sa.Index('ix_ft_symbol_tf_ts', 'symbol', 'timeframe', 'ts_utc')
    )

    op.create_table(
        'models',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String(64), nullable=False),
        sa.Column('version', sa.String(32), nullable=False),
        sa.Column('meta', sa.JSON, nullable=False),
        sa.Column('path', sa.String(256), nullable=False),
        sa.UniqueConstraint('name', 'version', name='uq_model_key')
    )

    op.create_table(
        'predictions',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('symbol', sa.String(16), nullable=False),
        sa.Column('timeframe', sa.String(8), nullable=False),
        sa.Column('ts_utc', sa.BigInteger, nullable=False),
        sa.Column('horizon', sa.Integer, nullable=False),
        sa.Column('q05', sa.Float, nullable=False),
        sa.Column('q50', sa.Float, nullable=False),
        sa.Column('q95', sa.Float, nullable=False),
        sa.Column('q05_c', sa.Float, nullable=True),
        sa.Column('q95_c', sa.Float, nullable=True),
        sa.Column('credibility', sa.Float, nullable=True),
        sa.Column('diag', sa.JSON, nullable=True),
        sa.UniqueConstraint('symbol', 'timeframe', 'ts_utc', 'horizon', name='uq_pred_key'),
        sa.Index('ix_pr_symbol_tf_ts_h', 'symbol', 'timeframe', 'ts_utc', 'horizon')
    )

    op.create_table(
        'residuals',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('symbol', sa.String(16), nullable=False),
        sa.Column('timeframe', sa.String(8), nullable=False),
        sa.Column('ts_utc', sa.BigInteger, nullable=False),
        sa.Column('horizon', sa.Integer, nullable=False),
        sa.Column('residual', sa.Float, nullable=False),
        sa.Index('ix_rs_symbol_tf_ts_h', 'symbol', 'timeframe', 'ts_utc', 'horizon')
    )

    op.create_table(
        'signals',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('symbol', sa.String(16), nullable=False),
        sa.Column('timeframe', sa.String(8), nullable=False),
        sa.Column('ts_utc', sa.BigInteger, nullable=False),
        sa.Column('p_hit', sa.Float, nullable=False),
        sa.Column('rr', sa.Float, nullable=False),
        sa.Column('expectancy', sa.Float, nullable=False),
        sa.Column('credibility', sa.Float, nullable=False),
        sa.Column('state', sa.String(8), nullable=False),  # GO_LIVE/HALT
        sa.Index('ix_sg_symbol_tf_ts', 'symbol', 'timeframe', 'ts_utc')
    )

    op.create_table(
        'reports',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('kind', sa.String(16), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('payload', sa.JSON, nullable=False)
    )


def downgrade() -> None:
    op.drop_table('reports')
    op.drop_table('signals')
    op.drop_table('residuals')
    op.drop_table('predictions')
    op.drop_table('models')
    op.drop_table('features')
    op.drop_table('market_data')


