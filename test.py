from ultralytics import YOLO

# Скачает yolov8n.pt автоматически (~6MB)
model = YOLO("yolov8n.pt")

# Запускаем детекцию напрямую по URL
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    save=True,
    conf=0.25
)

print("✅ Детекция завершена!")
print(f"Найдено объектов: {len(results[0].boxes)}")
for box in results[0].boxes:
    cls = results[0].names[int(box.cls)]
    conf = float(box.conf)
    print(f"  - {cls}: {conf:.0%}")
