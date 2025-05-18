from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("best.pt")
result = model.predict(source="0", show=True)