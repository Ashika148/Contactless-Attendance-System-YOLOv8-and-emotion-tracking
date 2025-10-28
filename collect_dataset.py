import cv2
import os

# Step 1: Ask for student name
name = input("Enter student name: ").strip()

# Step 2: Create folder for this student
folder = os.path.join("dataset", name)
os.makedirs(folder, exist_ok=True)

# Step 3: Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Step 4: Start webcam
cap = cv2.VideoCapture(0)
count = 0
max_images = 50  # Number of images per student

print("📷 Collecting dataset... Press 'q' to quit early")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw box around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (200, 200))

        if count < max_images:
            img_name = os.path.join(folder, f"{name}_{count}.jpg")
            cv2.imwrite(img_name, face_resized)
            print(f"✅ Saved {img_name}")
            count += 1
        else:
            print("🎉 Dataset collection complete!")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cv2.imshow("Collecting Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
