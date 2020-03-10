import cv2

# 导入人脸级联分类器引擎，'.xml'文件里包含训练出来的人脸特征，cv2.data.haarcascades即为存放所有级联分类器模型文件的目录
face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# 调用摄像头
cap=cv2.VideoCapture(0)

# 循环不断获取摄像头的影像
while True:
    #获取摄像头拍摄到的图像
    ret,frame=cap.read() # ret图像获取是否成功，frame就是img
    if not ret:
        print("没有捕捉到图像")
    img=frame
    # 获取每一张人脸的坐标
    faces = face.detectMultiScale(img, scaleFactor=1.1, minNeighbors=15)
    # 使用方框，把人脸框起来
    for x, y, w, h in faces:
        # 人脸框
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # 框选出人脸框，在人脸区域而不是全图中进行人眼检测，节省计算资源
        face_area = img[y:y + h, x:x + w]
        # 获取每一张人眼的坐标
        eyes = eye.detectMultiScale(face_area, scaleFactor=1.3, minNeighbors=25)
        # 使用方框，把人眼框起来
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

        # 框选出人脸框，在人脸区域而不是全图中进行笑容检测，节省计算资源
        # smile_area = img[y:y + h, x:x + w]
        # 获取每一张人眼的坐标
        smiles = smile.detectMultiScale(face_area, scaleFactor=1.3, minNeighbors=70,minSize=(25,25),flags=cv2.CASCADE_SCALE_IMAGE)
        # 使用方框，把人眼框起来
        for ex, ey, ew, eh in smiles:
            cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)
            cv2.putText(img,'smile',(x,y-7),3,1.2,(0,0,255),2,cv2.LINE_AA)
        # for sx, sy, sw, sh in smiles:
        #     cv2.rectangle(face_area, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)

        # 实时展示效果画面
        cv2.imshow("frame",img)
        # 每5毫秒监听一次键盘动作
        if cv2.waitKey(5) &0xFF == ord('q'):
            break

# 最后，关闭所有窗口
cap.release()
cv2.destroyWindows()