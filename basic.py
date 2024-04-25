from cv2 import cv2
import time
import mediapipe as mp


vid = cv2.VideoCapture("assets/Videos/Small Talk.mp4")

ptime = 0
ctime = time.time()

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=5, circle_radius=5)

while True :
    suc,frame = vid.read()

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(img_rgb)

    if results.multi_face_landmarks :
        for face_lm in results.multi_face_landmarks :
            mpDraw.draw_landmarks(frame,face_lm,mpFaceMesh.FACEMESH_CONTOURS,drawSpec,drawSpec)
            for id,lm in enumerate(face_lm.landmark) :
                fh , fw , fl = frame.shape
                x,y = int(fw*lm.x) , int(fh*lm.y)
                cv2.putText(frame,f'{id}',(x,y) ,cv2.FONT_HERSHEY_PLAIN , 3 , (50,50,50) , 2)


    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(frame,f'FPS:{int(fps)}',(50,150),cv2.FONT_HERSHEY_SIMPLEX, 5,(0,200,0), 8)
    frame = cv2.resize(frame , (1080,720))

    cv2.imshow("Video", frame)
    if cv2.waitKey(5) & 0xFF==ord('q') :
        break

cv2.F