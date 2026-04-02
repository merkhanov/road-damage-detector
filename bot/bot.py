"""
Telegram-бот для приёма фотографий повреждений дорог.
Пользователь отправляет фото -> бот анализирует через FastAPI -> возвращает результат.
Затем запрашивает геолокацию и сохраняет отчёт.
"""

import os
import logging
from io import BytesIO

import httpx
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")

logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

WAITING_PHOTO, WAITING_LOCATION = range(2)

SEVERITY_RU = {
    "none": "Не обнаружено",
    "low": "Низкая",
    "medium": "Средняя",
    "critical": "Критическая",
}

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def _api_headers() -> dict:
    headers = {}
    if API_SECRET_KEY:
        headers["X-API-Key"] = API_SECRET_KEY
    return headers


async def _geocode_address(address: str) -> tuple[float, float, str] | None:
    """Превращает адрес в координаты через Nominatim (OpenStreetMap)."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                NOMINATIM_URL,
                params={
                    "q": address,
                    "format": "json",
                    "limit": 1,
                    "addressdetails": 1,
                },
                headers={"User-Agent": "RoadDamageBot/1.0"},
            )
            response.raise_for_status()
            results = response.json()
    except Exception:
        logger.exception("Geocoding error")
        return None

    if not results:
        return None

    hit = results[0]
    return float(hit["lat"]), float(hit["lon"]), hit.get("display_name", address)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот для обнаружения повреждений дорог.\n\n"
        "Отправьте мне фото дороги, и я определю есть ли повреждения.\n"
        "Команды:\n"
        "/start - начать\n"
        "/cancel - отменить"
    )
    return WAITING_PHOTO


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Получаем фото, отправляем на анализ в FastAPI."""
    await update.message.reply_text("Анализирую фото...")

    photo = update.message.photo[-1]
    file = await photo.get_file()

    buf = BytesIO()
    await file.download_to_memory(buf)
    buf.seek(0)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/analyze",
                files={"file": ("photo.jpg", buf, "image/jpeg")},
                headers=_api_headers(),
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as e:
        logger.error("API error: %s - %s", e.response.status_code, e.response.text)
        await update.message.reply_text("Ошибка при анализе. Попробуйте позже.")
        return WAITING_PHOTO
    except Exception:
        logger.exception("Connection error")
        await update.message.reply_text(
            "Не удалось связаться с сервером. Убедитесь что backend запущен."
        )
        return WAITING_PHOTO

    context.user_data["last_result"] = data

    severity_text = SEVERITY_RU.get(data["severity"], data["severity"])
    confidence_pct = int(data["confidence"] * 100)

    if data["detected"]:
        text = (
            f"Повреждение обнаружено!\n"
            f"Серьёзность: {severity_text}\n"
            f"Уверенность: {confidence_pct}%\n\n"
            f"Укажите местоположение:\n"
            f"- Напишите адрес (например: ул. Абая 10, Алматы)\n"
            f"- Или отправьте геолокацию кнопкой ниже\n"
            f"- Или /skip чтобы сохранить без адреса"
        )
    else:
        text = (
            "Повреждений не обнаружено.\n"
            "Отправьте другое фото или /cancel для выхода."
        )
        await update.message.reply_text(text)
        return WAITING_PHOTO

    location_keyboard = ReplyKeyboardMarkup(
        [[KeyboardButton("Отправить геолокацию", request_location=True)]],
        one_time_keyboard=True,
        resize_keyboard=True,
    )
    await update.message.reply_text(text, reply_markup=location_keyboard)
    return WAITING_LOCATION


async def _save_report(update: Update, context: ContextTypes.DEFAULT_TYPE, lat: float, lng: float, address: str = ""):
    """Сохраняет отчёт через API."""
    data = context.user_data.get("last_result")
    if not data:
        await update.message.reply_text(
            "Нет данных анализа. Отправьте фото заново.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return WAITING_PHOTO

    report = {
        "latitude": lat,
        "longitude": lng,
        "severity": data["severity"],
        "confidence": data["confidence"],
        "image_path": data["annotated_image_url"],
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/report",
                json=report,
                headers=_api_headers(),
            )
            response.raise_for_status()
    except Exception:
        logger.exception("Failed to save report")
        await update.message.reply_text(
            "Не удалось сохранить отчёт. Попробуйте позже.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return WAITING_PHOTO

    location_info = address if address else f"{lat:.4f}, {lng:.4f}"
    await update.message.reply_text(
        f"Отчёт сохранён!\n"
        f"Адрес: {location_info}\n"
        "Отправьте ещё фото или /cancel для выхода.",
        reply_markup=ReplyKeyboardRemove(),
    )

    context.user_data.pop("last_result", None)
    return WAITING_PHOTO


async def handle_location(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Получаем геолокацию (кнопка на телефоне) и сохраняем отчёт."""
    location = update.message.location
    return await _save_report(update, context, location.latitude, location.longitude)


async def handle_text_location(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Получаем адрес текстом, геокодим через Nominatim."""
    text = update.message.text.strip()

    await update.message.reply_text("Ищу адрес...")

    result = await _geocode_address(text)
    if not result:
        await update.message.reply_text(
            "Адрес не найден. Попробуйте написать точнее:\n"
            "Например: ул. Абая 10, Алматы"
        )
        return WAITING_LOCATION

    lat, lng, display_name = result
    return await _save_report(update, context, lat, lng, display_name)


async def handle_skip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Сохраняет отчёт без координат (0, 0)."""
    return await _save_report(update, context, 0.0, 0.0, "Не указан")


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Отменено. Отправьте /start чтобы начать заново.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


def main():
    if not TELEGRAM_TOKEN:
        raise ValueError(
            "TELEGRAM_TOKEN не задан.\n"
            "Создайте .env файл с TELEGRAM_TOKEN=ваш_токен"
        )

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            WAITING_PHOTO: [
                MessageHandler(filters.PHOTO, handle_photo),
            ],
            WAITING_LOCATION: [
                MessageHandler(filters.LOCATION, handle_location),
                MessageHandler(filters.PHOTO, handle_photo),
                CommandHandler("skip", handle_skip),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_location),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(conv)

    logger.info("Бот запущен. Ожидание сообщений...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
