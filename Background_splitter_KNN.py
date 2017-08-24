import cv2
import numpy as np

camera = cv2.VideoCapture(0) # 参数0表示第一个摄像头
bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
while True:
    grabbed, frame_lwpCV = camera.read()
    fgmask = bs.apply(frame_lwpCV) # 背景分割器，该函数计算了前景掩码
    # 二值化阈值处理，前景掩码含有前景的白色值以及阴影的灰色值，在阈值化图像中，将非纯白色（244~255）的所有像素都设为0，而不是255
    th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    dilated = cv2.dilate(th, es, iterations=2) # 形态学膨胀
    image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 该函数计算一幅图像中目标的轮廓
    for c in contours:
        if cv2.contourArea(c) > 1600:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (255, 0, 255), 2)

    cv2.imshow('mog', fgmask)
    cv2.imshow('thresh', th)
    cv2.imshow('detection', frame_lwpCV)
    key = cv2.waitKey(1) & 0xFF
    # 按'q'健退出循环
    if key == ord('q'):
        break
# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()