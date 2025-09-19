from ultralytics import YOLO
import cv2

# le modèle entraîné
model = YOLO(r"C:\Users\lenovo\Desktop\test\best.pt")

# Charger l'image a tester 
image_path = r"C:\Users\lenovo\Desktop\test\farahab.jpg" 
image = cv2.imread(image_path)

# Faire la détection telque results est sous forme de liste c porquoi on ecrit results[0]car on a une seule image 
results = model(image)

# Annoter l'image (dessiner les boîtes telque l'objet results a une methode plot t3awadh cv2.rectangle et cv2.puttext)
annotated_image = results[0].plot()

# Afficher le résultat
cv2.imshow("Détection de Crabe", annotated_image)

# Sauvegarder l'image annotée 
cv2.imwrite("resultat_detection.jpg", annotated_image)

# Attendre qu'on appuie sur une touche pour fermer
cv2.waitKey(0)
cv2.destroyAllWindows()
