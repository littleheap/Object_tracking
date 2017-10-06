# 【Python+OpenCV】实现检测场景内是否有物体移动，并进行人脸检测抓拍
import cv2
import time

save_path = './face/'
# 人脸检测模型
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
# 眼睛检测模型
eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')

# 读取摄像头视频流
camera = cv2.VideoCapture(0)

# 帧率
fps = 5
# 前一帧记录
pre_frame = None
# 打印图片数量累计
i = 0

# 死循环不断读取视频帧
while True:
    # 记录开始时间
    start = time.time()
    # 读取视频流
    grabbed, screen = camera.read()
    # 转灰度图
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    # 如果grabbed为否则停止识别
    if not grabbed:
        break

    # 记录停止时间
    end = time.time()

    # 用模型进行人脸检测
    faces = face_cascade.detectMultiScale(screen_gray, 1.3, 5)

    # 遍历所有检测到的人脸
    for (x, y, w, h) in faces:
        # 勾勒轮廓
        cv2.rectangle(screen, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # 检出人脸区域后，取上半部分，因为眼睛在上边啊，这样精度会高一些
        # 截取灰度图片识别的上半部分
        roi_screen_gray = screen_gray[y:int(y + h / 2), x:x + w]
        # 截取原始图片的上半部分
        roi_screen_frame = screen[y:int(y + h / 2), x:x + w]
        # 将检测到的人脸写入文件
        cv2.imwrite(save_path + str(i) + '.jpg', screen[y:y + h, x:x + w])
        # 累加1
        i += 1

        # 用模型进行眼部识别
        eyes = eye_cascade.detectMultiScale(roi_screen_gray, 1.03, 5)
        # 遍历所有眼部识别
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_screen_frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    # 显示
    cv2.imshow('原图', screen)

    # 运动检测部分

    # 获取起止时间
    seconds = end - start
    # 如果截取时间比5帧率快
    if seconds < 1.0 / fps:
        # 睡到5帧的频率，之后再往下进行
        time.sleep(1.0 / fps - seconds)

    # 将灰度图规范化
    screen_gray = cv2.resize(screen_gray, (500, 500))
    '''
        用高斯滤波进行模糊处理，进行处理的原因：
        每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。
        对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
    '''
    screen_gray = cv2.GaussianBlur(screen_gray, (21, 21), 0)
    '''
        在完成对帧的灰度转换和平滑后，就可计算与背景帧的差异，并得到一个差分图（different map）。
        还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，
        从而对孔（hole）和缺陷（imperfection）进行归一化处理
    '''
    if pre_frame is None:
        pre_frame = screen_gray
    else:
        # 计算当前画面与前一帧之间的差分图
        img_delta = cv2.absdiff(pre_frame, screen_gray)
        # 设置优化阈值
        _, thresh = cv2.threshold(img_delta, 200, 255, cv2.THRESH_BINARY)
        # 膨胀处理，使画面圆滑
        thresh = cv2.dilate(thresh, None, iterations=2)
        # 获取检测边缘坐标数据
        _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 遍历所有检测边缘
        for c in contours:
            # 设置敏感度，小于2000面积的运动区域不显示
            if cv2.contourArea(c) < 1000:
                continue
            else:
                # 一旦检测到东西动，就打印这句话
                print("咦,有什么东西在动0.0！")
                break
        # 将当前帧设置为下一阵的背景
        pre_frame = screen_gray

    # 设置键盘等待指令
    key = cv2.waitKey(1) & 0xFF
    # 按'q'键退出循环
    if key == ord('q'):
        break

# 关闭摄像头
camera.release()
# 关闭所有窗口
cv2.destroyAllWindows()
