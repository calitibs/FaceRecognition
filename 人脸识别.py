import cv2

# 读取图片,显示形式：0灰色 1彩色
img = cv2.imread('demo/p7.jpg',1)

# 获取人脸位置(他的引擎就是级联分类器中的按.xml文件匹配的人脸检测模型)
# 路径 E:\python\IDLE\Lib\site-packages\cv2\data
locations = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 获取每一张人脸的坐标
# scaleFactor：缩放比例  minNeighbors：临近的检测次数，数值越大，检测要求越高
faces = locations.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)

# 使用方框，把人脸框起来
for x, y, w, h in faces:
    # 左下角的x，y以及宽度，高度
    img=cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0, 0), 2)

# 展示
cv2.imshow('image', img)

# 保存图片
cv2.imwrite('result/result011.jpg', img)  # wait for 's' key to save and exit

# 等待键盘事件
cv2.waitKey()

# 销毁窗口
cv2.destroyAllWindows()


