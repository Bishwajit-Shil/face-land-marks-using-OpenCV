import cv2
import mediapipe as mp
import time

img = cv2.imread("kat.jpg",cv2.IMREAD_COLOR)

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh 
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=3)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
 

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = faceMesh.process(imgRGB)
if results.multi_face_landmarks:
    for faceLms in results.multi_face_landmarks:
        mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
                                drawSpec,drawSpec)
        # for id,lm in enumerate(faceLms.landmark):
        #     #print(lm)
        #     ih, iw, ic = img.shape
        #     x,y = int(lm.x*iw), int(lm.y*ih)
        #     # print(id,x,y)
        #     cv2.putText(img,f'{id}',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)


img = cv2.resize(img, (640,480))
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
