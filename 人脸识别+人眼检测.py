import cv2

img=cv2.imread('demo/p1.jpg')

# 导入人脸级联分类器引擎，'.xml'文件里包含训练出来的人脸特征，cv2.data.haarcascades即为存放所有级联分类器模型文件的目录
face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


# 获取每一张人脸的坐标
faces = face.detectMultiScale(img, scaleFactor=1.1, minNeighbors=15)

# 使用方框，把人脸框起来
for x, y, w, h in faces:
    # 人脸框
    img=cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0, 0), 2)
    # 框选出人脸框，在人脸区域而不是全图中进行人眼检测，节省计算资源
    face_area=img[y:y+h,x:x+w]
    # 获取每一张人眼的坐标
    eyes=eye.detectMultiScale(face_area, scaleFactor=1.5, minNeighbors=1)
    # 使用方框，把人眼框起来
    for ex,ey,ew,eh in eyes:
        cv2.rectangle(face_area,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)


cv2.imshow("face&eye",img)
cv2.waitKey(0)
cv2.destroyAllWindows()