import cv2
from ultralytics import YOLO
import time

# Charger le modèle
model = YOLO("best.pt")

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    # Passer l'image dans le modèle
    results = model.predict(source=frame, save=False, conf=0.6, verbose=False)

    # Dessiner les boîtes
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    end = time.time()
    fps = 1 / (end - start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Afficher le résultat
    cv2.imshow("YOLOv8 Detection", frame)

    # Quitter si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#pour fermer proprement le webcam
cap.release()
#fermer toute les fenetres cree par opencv 
cv2.destroyAllWindows()


