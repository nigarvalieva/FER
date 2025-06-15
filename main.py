import cv2
from ultralytics import YOLO
import signal
import csv
import json
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

# Загружаем модель
model = YOLO("best.pt")

# Список эмоций (из твоей модели, адаптируй при необходимости)
emotion_labels_list = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Списки для хранения данных
timestamps = []
emotion_predictions = []

# Обработка Ctrl+C
running = True
def signal_handler(sig, frame):
    global running
    print("\nStopped by user (CTRL+C pressed).")
    running = False
signal.signal(signal.SIGINT, signal_handler)

# Видеозахват
cap = cv2.VideoCapture(0)

print("Starting emotion recording... Press Ctrl+C to stop.")

while running:
    ret, frame = cap.read()
    if not ret:
        break

    # Предсказание
    results = model.predict(source=frame, conf=0.25, verbose=False)

    # Аннотированный фрейм для отображения
    annotated_frame = results[0].plot()
    cv2.imshow("Real-Time Emotion Recognition", annotated_frame)

    # Получаем метки
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if 0 <= cls_id < len(emotion_labels_list):
                emotion = emotion_labels_list[cls_id]
                timestamps.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                emotion_predictions.append(emotion)

    # Выход по клавише
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped by user (key 'q' pressed).")
        break

cap.release()
cv2.destroyAllWindows()

# Сохраняем в CSV
with open("emotion_log.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Emotion"])
    for ts, emo in zip(timestamps, emotion_predictions):
        writer.writerow([ts, emo])

# Сохраняем в JSON
with open("emotion_log.json", "w", encoding="utf-8") as f:
    json.dump([{"Timestamp": ts, "Emotion": emo} for ts, emo in zip(timestamps, emotion_predictions)], f, ensure_ascii=False, indent=4)

print("Emotion data saved.")

# Сводная статистика
summary = defaultdict(int)
for emo in emotion_predictions:
    summary[emo] += 1

total = len(emotion_predictions)
print("\nSummary Statistics:")
for emo in emotion_labels_list:
    count = summary[emo]
    percent = round((count / total) * 100, 2) if total > 0 else 0
    print(f"{emo}: {count} ({percent}%)")

# График: Эмоции по времени
emotion_to_num = {emo: i for i, emo in enumerate(emotion_labels_list)}
y_vals = [emotion_to_num[emo] for emo in emotion_predictions]
plt.figure(figsize=(10, 3))
plt.step(range(len(y_vals)), y_vals, where='post')
plt.yticks(list(emotion_to_num.values()), list(emotion_to_num.keys()))
plt.xlabel("Frame Index")
plt.ylabel("Emotion")
plt.title("Emotion Timeline During Gameplay")
plt.tight_layout()
plt.savefig("emotion_timeline.png")
plt.close()

# Гистограмма: Распределение эмоций
plt.figure(figsize=(6, 4))
plt.bar(summary.keys(), summary.values(), color="lightgreen")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.title("Emotion Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("emotion_distribution.png")
plt.close()

print("Graphs saved.") 
