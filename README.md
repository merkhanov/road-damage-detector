# Road Damage Detector

AI-система для автоматического обнаружения повреждений дорог (ям, трещин) с помощью Computer Vision.

## Стек
- **YOLOv8** — детекция повреждений (7 классов)
- **FastAPI** — backend API
- **PostgreSQL** — хранение отчётов
- **Telegram бот** — приём фото от граждан
- **Next.js + Leaflet.js** — дашборд с картой

## Классы повреждений
| Код | Тип |
|-----|-----|
| D00 | Продольная трещина |
| D10 | Поперечная трещина |
| D20 | Аллигаторная трещина |
| D40 | Яма |
| D43 | Повреждённый переход |
| D44 | Повреждённая разметка |
| D50 | Люк |

## Быстрый старт

### 1. Установка зависимостей
```bash
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Скачать датасет
```bash
pip install roboflow
python data/download.py
```

### 3. Обучить модель
```bash
python train.py
```
Веса сохранятся в `backend/best.pt`.

### 4. Настроить окружение
```bash
cp .env.example .env
# Отредактируйте .env: DATABASE_URL, TELEGRAM_TOKEN
```

### 5. Создать базу данных
```bash
createdb road_damage
```

### 6. Запустить backend
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
API доступен на http://localhost:8000/docs

### 7. Запустить Telegram-бота
```bash
python -m bot.bot
```

## API Эндпоинты
| Метод | URL | Описание |
|-------|-----|----------|
| POST | `/analyze` | Загрузить фото → получить результат детекции |
| POST | `/report` | Сохранить отчёт с координатами |
| GET | `/reports` | Получить все отчёты |
| GET | `/health` | Статус сервера |
