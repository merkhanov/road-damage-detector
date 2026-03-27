# Road Damage Detector

AI-система для автоматического обнаружения повреждений дорог (ям, трещин) с помощью Computer Vision.

## Стек
- YOLOv8 — детекция повреждений
- FastAPI — backend API
- PostgreSQL — хранение отчётов
- Next.js + Leaflet.js — дашборд с картой
- Telegram бот — приём фото от граждан

## Запуск
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```
