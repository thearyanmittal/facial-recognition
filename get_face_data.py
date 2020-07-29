import cv2
import os


name = input("Enter your name: \n")
final_num_pics = 50
num_pics = 50
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR, 'faces')
face_dir = os.path.join(img_dir, name.lower())

os.mkdir(face_dir)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("Face Data Collection", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=4)
    
    if num_pics > 0 and len(faces) > 0:
        for (x, y, w, h) in faces:
            roi = frame[y-100:y+h+100, x-100:x+w+100]
            cv2.imwrite(os.path.join(face_dir, f"{num_pics}.png"), roi)
        
        num_pics -= 1
        print(f"Collection Status: {int(((final_num_pics - num_pics) / final_num_pics) * 100)}% completed")


    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()