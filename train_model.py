
import os
import cv2
import pickle
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from sklearn.svm import SVC

dataset_path = "dataset"
cache_file = "embeddings.npz"

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

label_ids = {}
current_id = 0
embeddings = []
labels = []

if os.path.exists(cache_file):
    print("⚡ Loading cached embeddings...")
    data = np.load(cache_file, allow_pickle=True)
    embeddings = data["embeddings"].tolist()
    labels = data["labels"].tolist()
    label_ids = data["label_ids"].item()
else:
    print("📷 Extracting embeddings...")

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(("png", "jpg", "jpeg")):
                path = os.path.join(root, file)
                label = os.path.basename(root)
                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]

                img = cv2.imread(path)
                results = model(img, verbose=False)

                for box in results[0].boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box[:4])
                    face = img[y1:y2, x1:x2]
                    try:
                        embedding = DeepFace.represent(face, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
                        embeddings.append(embedding)
                        labels.append(id_)
                    except:
                        continue

    np.savez(cache_file, embeddings=embeddings, labels=labels, label_ids=label_ids)
    print(f"💾 Cached embeddings saved to {cache_file}")

X = np.array(embeddings)
y = np.array(labels)

# Train SVM with probability
clf = SVC(kernel="linear", probability=True)
clf.fit(X, y)

with open("face_classifier.pkl", "wb") as f:
    pickle.dump((clf, label_ids), f)

print("✅ Training complete! Model saved as 'face_classifier.pkl'")
print("Labels:", {v: k for k, v in label_ids.items()})
