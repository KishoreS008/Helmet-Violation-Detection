import os
import cv2
from ultralytics import YOLO
from datetime import datetime

# ✅ Load your pre-trained model
model = YOLO("best.pt")

# ✅ Show the model’s class names
print("Model classes:", model.names)

# ✅ Create a folder to store violations
if not os.path.exists("violations"):
    os.makedirs("violations")

# ✅ Start webcam
cap = cv2.VideoCapture(0)
last_saved = datetime.now()

while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("⚠️ Warning: Empty frame received from webcam")
        continue

    # Run detection
    results = model(frame)[0]
    status = "Safe"

    for r in results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        label = model.names[int(r.cls[0])]
        confidence = float(r.conf[0])

        # ✅ Normalize label for flexible comparison
        clean_label = label.strip().lower().replace("-", "").replace("_", "")
        print(f"Detected: {label} (clean: {clean_label}, confidence: {confidence:.2f})")

        # ✅ Calculate bounding box area
        box_area = (x2 - x1) * (y2 - y1)

        # ✅ Check for 'nonhelmet' and only if confidence & area are strong
        if "nonhelmet" in clean_label and confidence > 0.4 and box_area > 10000:
            status = "Violation"
            now = datetime.now()
            if (now - last_saved).total_seconds() > 2:
                filename = f"violations/violation_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[⚠️] Violation saved: {filename}")
                last_saved = now
            color = (0, 0, 255)  # Red for violation
        else:
            color = (0, 255, 0)  # Green box if helmet

        # ✅ Draw the detection box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ✅ Show status (Safe / Violation)
    cv2.putText(frame, f"Status: {status}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 0, 255) if status == "Violation" else (0, 255, 0), 3)

    # Show frame
    cv2.imshow("Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
