# Road Damage Detector

AI-система для автоматического обнаружения повреждений дорог на основе Computer Vision.
Граждане отправляют фото через Telegram-бот, система определяет тип и серьёзность повреждения, а отчёты сохраняются в базу данных с привязкой к адресу.

> **Decentrathon 5.0** — хакатон-проект, дедлайн 5 апреля 2026

## Архитектура

```
Пользователь
    │
    ▼
Telegram-бот ──► FastAPI backend ──► PostgreSQL
    │                  │
    │                  ▼
    │             YOLOv8 модель
    │                  │
    ◄──────────────────┘
   Ответ с результатом
```

## Стек технологий

| Компонент | Технология |
|-----------|------------|
| Модель | YOLOv8n (Ultralytics) |
| Backend API | FastAPI + Uvicorn |
| База данных | PostgreSQL + SQLAlchemy |
| Telegram-бот | python-telegram-bot + httpx |
| Геокодинг | Nominatim (OpenStreetMap) |

## Классы повреждений

| Код | Тип повреждения |
|-----|-----------------|
| D00 | Продольная трещина |
| D10 | Поперечная трещина |
| D20 | Аллигаторная трещина |
| D40 | Яма |
| D43 | Повреждённый переход |
| D44 | Повреждённая разметка |
| D50 | Люк |

## Уровни серьёзности

| Уровень | Уверенность модели | Описание |
|---------|-------------------|----------|
| low | < 40% | Незначительное повреждение |
| medium | 40–70% | Среднее повреждение |
| critical | > 70% | Критическое повреждение |

## Быстрый старт

### 1. Клонирование и установка

```bash
git clone https://github.com/<your-repo>/road-damage-detector.git
cd road-damage-detector
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Скачать датасет

```bash
pip install roboflow
python data/download.py
```

Для скачивания нужен `ROBOFLOW_API_KEY` в файле `.env`.

### 3. Обучить модель

```bash
python train.py
```

Обучение: 50 эпох, `imgsz=640`, YOLOv8n.
Лучшие веса сохраняются в `backend/best.pt`.

### 4. Настроить окружение

```bash
cp .env.example .env
```

Заполните `.env`:

| Переменная | Описание |
|------------|----------|
| `DATABASE_URL` | PostgreSQL connection string |
| `TELEGRAM_TOKEN` | Токен от @BotFather |
| `API_BASE_URL` | URL backend (по умолчанию `http://localhost:8000`) |
| `ROBOFLOW_API_KEY` | Ключ Roboflow для скачивания датасета |
| `API_SECRET_KEY` | Ключ для защиты API (опционально) |
| `ALLOWED_ORIGINS` | CORS-домены через запятую |

### 5. Создать базу данных

```bash
createdb road_damage
```

### 6. Запустить backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI: http://localhost:8000/docs

### 7. Запустить Telegram-бота

```bash
python -m bot.bot
```

## API эндпоинты

| Метод | URL | Описание |
|-------|-----|----------|
| `POST` | `/analyze` | Загрузить фото — получить результат детекции |
| `POST` | `/report` | Сохранить отчёт с координатами |
| `GET` | `/reports` | Получить все отчёты (лимит 500) |
| `GET` | `/health` | Статус сервера и модели |

### POST /analyze

Принимает изображение (jpg, png, webp, до 10 МБ), возвращает:

```json
{
  "detected": true,
  "severity": "critical",
  "confidence": 0.87,
  "annotated_image_url": "/static/results/photo.jpg",
  "detections": [
    {
      "class": "D40",
      "class_ru": "Яма",
      "confidence": 0.87,
      "severity": "critical",
      "bbox": [100, 200, 300, 400]
    }
  ],
  "timestamp": "2026-03-27T12:00:00"
}
```

### POST /report

```json
{
  "latitude": 43.238,
  "longitude": 76.945,
  "severity": "critical",
  "confidence": 0.87,
  "image_path": "/static/results/photo.jpg"
}
```

## Telegram-бот

Сценарий работы:

1. Пользователь отправляет `/start`
2. Отправляет фото повреждённой дороги
3. Бот анализирует и возвращает результат (тип, серьёзность, уверенность)
4. Просит указать адрес — три варианта:
   - Написать адрес текстом (например: `ул. Абая 10, Алматы`) — бот геокодирует через Nominatim
   - Отправить геолокацию кнопкой (на мобильном)
   - `/skip` — сохранить без адреса
5. Отчёт сохраняется в базу данных

## Безопасность

- API-ключ для защиты эндпоинтов (`X-API-Key` header)
- Rate limiting: 30 запросов/мин на IP
- Валидация файлов: размер (10 МБ) и расширение
- Валидация координат и входных данных
- CORS с ограниченным списком доменов
- Секреты через `.env` (не коммитятся)

## Структура проекта

```
road-damage-detector/
├── backend/
│   ├── main.py          # FastAPI приложение
│   ├── model.py         # YOLOv8 обёртка
│   ├── database.py      # SQLAlchemy + модель Report
│   ├── requirements.txt # Зависимости
│   └── best.pt          # Веса модели (после обучения)
├── bot/
│   └── bot.py           # Telegram-бот
├── data/
│   └── download.py      # Скачивание датасета с Roboflow
├── train.py             # Скрипт обучения
├── .env.example         # Шаблон переменных окружения
└── .gitignore
```

## Лицензия

MIT
