import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load cached embeddings
data = np.load("embeddings.npz", allow_pickle=True)
X = np.array(data["embeddings"].tolist())
y = np.array(data["labels"].tolist())
label_ids = data["label_ids"].item()
id_to_label = {v: k for k, v in label_ids.items()}

# Load trained classifier
with open("face_classifier.pkl", "rb") as f:
    clf, saved_label_ids = pickle.load(f)

print("🔍 Evaluating model...")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train again (fresh) on train set
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc*100:.2f}%")
print("\n📊 Classification Report:\n", classification_report(
    y_test, y_pred, target_names=[id_to_label[i] for i in sorted(set(y))]
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[id_to_label[i] for i in sorted(set(y))],
            yticklabels=[id_to_label[i] for i in sorted(set(y))])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Accuracy Bar Graph
plt.figure(figsize=(5, 4))
plt.bar(["Accuracy"], [acc*100], color="green")
plt.ylim(0, 100)
plt.title("Model Accuracy (%)")
plt.show()



