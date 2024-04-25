import cv2
import mediapipe as mp
import time
 
cap = cv2.VideoCapture("assets/Videos/video.mp4")
pTime = 0
 
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh 
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=2, circle_radius=3)
 
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
                                  drawSpec,drawSpec)
            for id,lm in enumerate(faceLms.landmark):
                #print(lm)
                ih, iw, ic = img.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                print(id,x,y)
                # cv2.putText(img,f'{id}',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
 
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (img.shape[1]-400,img.shape[0]-100), cv2.FONT_HERSHEY_PLAIN,
                7, (255, 255, 0), 7)
    img = cv2.resize(img, (720,640))
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF==ord('q') : 
        break

cap.release()
cv2.destroyAllWindows()