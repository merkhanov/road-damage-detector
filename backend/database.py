"""
Настройка SQLAlchemy + модель таблицы reports.
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/road_damage"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    severity = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    image_path = Column(String(500), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Создаёт таблицы, если их ещё нет."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Генератор сессии для FastAPI Depends."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
