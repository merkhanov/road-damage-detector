"""
Скрипт обучения YOLOv8 на датасете Road Damage Detection.
Классы: D00 (продольные трещины), D10 (поперечные), D20 (аллигаторные),
        D40 (ямы), D43 (повреждённый переход), D44 (повреждённая разметка), D50 (люк)
"""

from pathlib import Path
from ultralytics import YOLO


DATA_YAML = Path("data/Road-damage-detection-2/data.yaml")
EPOCHS = 50
IMGSZ = 640
MODEL_BASE = "yolov8n.pt"
OUTPUT_WEIGHTS = Path("backend/best.pt")


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Датасет не найден: {DATA_YAML}\n"
            "Сначала запустите: python data/download.py"
        )

    print(f"Загрузка базовой модели: {MODEL_BASE}")
    model = YOLO(MODEL_BASE)

    print(f"Начинаем обучение: {EPOCHS} эпох, imgsz={IMGSZ}")
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=-1,       # автоподбор batch size под GPU/RAM
        patience=10,    # ранняя остановка если mAP не растёт
        save=True,
        project="runs/train",
        name="road_damage",
        exist_ok=True,
    )

    # Копируем лучшие веса в backend/
    best_src = Path("runs/train/road_damage/weights/best.pt")
    if best_src.exists():
        OUTPUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(best_src, OUTPUT_WEIGHTS)
        print(f"\nЛучшие веса сохранены: {OUTPUT_WEIGHTS}")
    else:
        print(f"\nОШИБКА: файл {best_src} не найден")
        return

    # Итоговые метрики
    metrics = model.val()
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
    print("=" * 50)
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
