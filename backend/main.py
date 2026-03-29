"""
FastAPI-сервер для анализа повреждений дорог.
Эндпоинты:
  POST /analyze  — загрузить фото, получить результат детекции
  POST /report   — сохранить отчёт с координатами в PostgreSQL
  GET  /reports  — получить все отчёты
"""

import os
import uuid
import shutil
import logging
import time
from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from backend.database import init_db, get_db, Report
from backend.model import RoadDamageModel

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Road Damage Detector API",
    description="API для обнаружения повреждений дорог с помощью YOLOv8",
    version="1.0.0",
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("backend/static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path("backend/static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
API_KEY = os.getenv("API_SECRET_KEY")

model: RoadDamageModel | None = None

# Rate limiter: IP -> list of timestamps
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_MAX = 30       # запросов
RATE_LIMIT_WINDOW = 60    # за N секунд


def _check_rate_limit(client_ip: str):
    now = time.time()
    timestamps = _rate_limit_store[client_ip]
    _rate_limit_store[client_ip] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(429, "Слишком много запросов. Попробуйте позже.")
    _rate_limit_store[client_ip].append(now)


def _verify_api_key(request: Request):
    """Проверяет API-ключ, если он задан в .env."""
    if not API_KEY:
        return
    key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if key != API_KEY:
        raise HTTPException(403, "Неверный API-ключ")


@app.on_event("startup")
def startup():
    global model
    init_db()
    try:
        model = RoadDamageModel()
        logger.info("Модель загружена успешно")
    except FileNotFoundError as e:
        logger.warning(f"{e}")
        logger.warning("Сервер запущен без модели. Сначала обучите: python train.py")


# ── Схемы ─────────────────────────────────────────────

class SeverityLevel(str, Enum):
    none = "none"
    low = "low"
    medium = "medium"
    critical = "critical"


class AnalyzeResponse(BaseModel):
    detected: bool
    severity: str
    confidence: float
    annotated_image_url: str
    detections: list
    timestamp: str


class ReportCreate(BaseModel):
    latitude: float
    longitude: float
    severity: SeverityLevel
    confidence: float
    image_path: str

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        if not -90 <= v <= 90:
            raise ValueError("Широта должна быть от -90 до 90")
        return v

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        if not -180 <= v <= 180:
            raise ValueError("Долгота должна быть от -180 до 180")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Уверенность должна быть от 0 до 1")
        return v


class ReportResponse(BaseModel):
    id: int
    latitude: float
    longitude: float
    severity: str
    confidence: float
    image_path: str
    timestamp: str

    class Config:
        from_attributes = True


# ── Эндпоинты ─────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request, file: UploadFile = File(...)):
    """Принимает изображение, запускает детекцию, возвращает результат."""
    _verify_api_key(request)
    _check_rate_limit(request.client.host)

    if model is None:
        raise HTTPException(503, "Модель не загружена")

    # Валидация расширения файла
    ext = Path(file.filename or "upload.jpg").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Неподдерживаемый формат. Разрешены: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Валидация размера файла
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, f"Файл слишком большой. Максимум: {MAX_FILE_SIZE // (1024*1024)} МБ")

    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / filename

    with open(save_path, "wb") as f:
        f.write(contents)

    try:
        result = model.predict(save_path)
    except Exception:
        logger.exception("Ошибка при детекции")
        save_path.unlink(missing_ok=True)
        raise HTTPException(500, "Ошибка при анализе изображения")

    annotated = Path(result.annotated_image_path)
    if annotated.exists():
        try:
            relative = annotated.relative_to(STATIC_DIR)
            annotated_url = f"/static/{relative}"
        except ValueError:
            annotated_url = f"/static/uploads/{filename}"
    else:
        annotated_url = f"/static/uploads/{filename}"

    return AnalyzeResponse(
        detected=result.detected,
        severity=result.severity,
        confidence=result.confidence,
        annotated_image_url=annotated_url,
        detections=result.detections,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/report", response_model=ReportResponse)
def create_report(report: ReportCreate, request: Request, db: Session = Depends(get_db)):
    """Сохраняет отчёт о повреждении в базу данных."""
    _verify_api_key(request)
    _check_rate_limit(request.client.host)

    db_report = Report(
        latitude=report.latitude,
        longitude=report.longitude,
        severity=report.severity.value,
        confidence=report.confidence,
        image_path=report.image_path,
        timestamp=datetime.utcnow(),
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)

    return ReportResponse(
        id=db_report.id,
        latitude=db_report.latitude,
        longitude=db_report.longitude,
        severity=db_report.severity,
        confidence=db_report.confidence,
        image_path=db_report.image_path,
        timestamp=db_report.timestamp.isoformat(),
    )


@app.get("/reports", response_model=list[ReportResponse])
def get_reports(request: Request, db: Session = Depends(get_db)):
    """Возвращает все отчёты из базы данных."""
    _verify_api_key(request)

    reports = db.query(Report).order_by(Report.timestamp.desc()).limit(500).all()
    return [
        ReportResponse(
            id=r.id,
            latitude=r.latitude,
            longitude=r.longitude,
            severity=r.severity,
            confidence=r.confidence,
            image_path=r.image_path,
            timestamp=r.timestamp.isoformat(),
        )
        for r in reports
    ]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }
