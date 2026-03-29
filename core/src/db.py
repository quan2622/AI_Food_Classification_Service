from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, String, Text, create_engine, func, select, text
from sqlalchemy import event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from config import DATABASE_URL


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = 'prediction_logs'

    image_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    original_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    stored_path: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    top1_class_name: Mapped[str] = mapped_column(String(255), nullable=False)
    top1_confidence: Mapped[float] = mapped_column(nullable=False)
    predictions: Mapped[list] = mapped_column(JSON, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    is_feedback_received: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    confirmed_label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)


parsed_url = make_url(DATABASE_URL)
query = dict(parsed_url.query)
DB_SCHEMA = query.pop('schema', None)
clean_database_url = parsed_url.set(query=query).render_as_string(hide_password=False)

engine = create_engine(clean_database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


if DB_SCHEMA:
    @event.listens_for(engine, 'connect')
    def set_search_path(dbapi_connection, connection_record):
        del connection_record
        with dbapi_connection.cursor() as cursor:
            cursor.execute(f'SET search_path TO "{DB_SCHEMA}"')


def init_db():
    Base.metadata.create_all(bind=engine)


def check_db_connection():
    with engine.connect() as connection:
        connection.execute(text('SELECT 1'))
    return True


@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_prediction_log(payload):
    with session_scope() as session:
        prediction = PredictionLog(**payload)
        session.add(prediction)


def get_prediction_log(image_id):
    with session_scope() as session:
        statement = select(PredictionLog).where(PredictionLog.image_id == image_id)
        record = session.execute(statement).scalar_one_or_none()
        if record is None:
            raise FileNotFoundError(f'Không tìm thấy metadata cho image_id={image_id}')
        return {
            'image_id': record.image_id,
            'original_filename': record.original_filename,
            'stored_path': record.stored_path,
            'model_name': record.model_name,
            'top1_prediction': {
                'class_name': record.top1_class_name,
                'confidence': record.top1_confidence,
            },
            'predictions': record.predictions,
            'uploaded_at': record.uploaded_at.isoformat(),
            'is_feedback_received': record.is_feedback_received,
            'confirmed_label': record.confirmed_label,
            'is_correct': record.is_correct,
            'notes': record.notes,
            'reviewed_path': record.reviewed_path,
            'reviewed_at': record.reviewed_at.isoformat() if record.reviewed_at else None,
        }


def update_prediction_feedback(image_id, confirmed_label, is_correct, notes, reviewed_path):
    with session_scope() as session:
        statement = select(PredictionLog).where(PredictionLog.image_id == image_id)
        record = session.execute(statement).scalar_one_or_none()
        if record is None:
            raise FileNotFoundError(f'Không tìm thấy metadata cho image_id={image_id}')

        record.is_feedback_received = True
        record.confirmed_label = confirmed_label
        record.is_correct = is_correct
        record.notes = notes
        record.reviewed_path = reviewed_path
        record.reviewed_at = datetime.now()


def get_feedback_stats(class_names):
    with session_scope() as session:
        uploads_count = session.scalar(select(func.count()).select_from(PredictionLog)) or 0
        metadata_count = uploads_count

        reviewed_counts = {}
        for class_name in class_names:
            statement = select(func.count()).select_from(PredictionLog).where(
                PredictionLog.confirmed_label == class_name,
                PredictionLog.is_feedback_received.is_(True),
            )
            reviewed_counts[class_name] = session.scalar(statement) or 0

        return {
            'uploads_count': uploads_count,
            'reviewed_counts': reviewed_counts,
            'metadata_count': metadata_count,
        }
