import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise ValueError("ROBOFLOW_API_KEY не задан в .env")

rf = Roboflow(api_key=api_key)
project = rf.workspace("new-workspace-kj87b").project("road-damage-detection-iicdh")
dataset = project.version(2).download("yolov8")
