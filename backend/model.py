"""
Обёртка над YOLOv8 для детекции повреждений дорог.
Возвращает: обнаружено ли повреждение, серьёзность, уверенность, путь к аннотированному изображению.
"""

from pathlib import Path
from dataclasses import dataclass
from ultralytics import YOLO

# Маппинг кодов классов → человекочитаемые названия
CLASS_NAMES_RU = {
    "D00": "Продольная трещина",
    "D10": "Поперечная трещина",
    "D20": "Аллигаторная трещина",
    "D40": "Яма",
    "D43": "Повреждённый переход",
    "D44": "Повреждённая разметка",
    "D50": "Люк",
}

SEVERITY_EMOJI = {
    "none": "⚪",
    "low": "🟡",
    "medium": "🟠",
    "critical": "🔴",
}

WEIGHTS_PATH = Path(__file__).parent / "best.pt"
STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class PredictionResult:
    detected: bool
    severity: str          # "none" | "low" | "medium" | "critical"
    confidence: float
    annotated_image_path: str
    detections: list       # список отдельных детекций


def _severity_from_confidence(conf: float) -> str:
    if conf < 0.4:
        return "low"
    elif conf <= 0.7:
        return "medium"
    return "critical"


class RoadDamageModel:
    def __init__(self, weights: str | Path | None = None):
        path = Path(weights) if weights else WEIGHTS_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Веса модели не найдены: {path}\n"
                "Сначала обучите модель: python train.py"
            )
        self.model = YOLO(str(path))
        STATIC_DIR.mkdir(parents=True, exist_ok=True)

    def predict(self, image_path: str | Path, conf_threshold: float = 0.25) -> PredictionResult:
        """Запускает детекцию на изображении и возвращает результат."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")

        results = self.model.predict(
            source=str(image_path),
            conf=conf_threshold,
            save=True,
            project=str(STATIC_DIR),
            name="results",
            exist_ok=True,
        )

        result = results[0]
        boxes = result.boxes

        if len(boxes) == 0:
            annotated_name = image_path.name
            annotated_path = STATIC_DIR / "results" / annotated_name
            return PredictionResult(
                detected=False,
                severity="none",
                confidence=0.0,
                annotated_image_path=str(annotated_path),
                detections=[],
            )

        detections = []
        max_conf = 0.0
        for box in boxes:
            conf = float(box.conf)
            cls_id = int(box.cls)
            cls_name = result.names[cls_id]
            detections.append({
                "class": cls_name,
                "class_ru": CLASS_NAMES_RU.get(cls_name, cls_name),
                "confidence": round(conf, 3),
                "severity": _severity_from_confidence(conf),
                "bbox": box.xyxy[0].tolist(),
            })
            max_conf = max(max_conf, conf)

        annotated_name = image_path.name
        annotated_path = STATIC_DIR / "results" / annotated_name

        return PredictionResult(
            detected=True,
            severity=_severity_from_confidence(max_conf),
            confidence=round(max_conf, 3),
            annotated_image_path=str(annotated_path),
            detections=detections,
        )
