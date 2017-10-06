import cv2

# 获取第一个摄像头实时视频
camera = cv2.VideoCapture(0)
# 以kNN格式进行背景分割，detectShadows=True，表示检测阴影，反之不检测阴影
bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
# 膨胀结构元素
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# 死循环不断获取摄像头读取到的帧数
while True:
    # 获取每一帧图片
    _, screen = camera.read()
    # 用kNN算法进行背景分割
    process = bs.apply(screen)
    # 阈值设定，将非纯白色的所有像素都设为0（黑色），而不是255
    _, threshold = cv2.threshold(process.copy(), 180, 255, cv2.THRESH_BINARY)
    # 为了使效果更好，进行一次膨胀
    dilated = cv2.dilate(threshold, es, iterations=2)
    # 检测轮廓
    _, contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    # 一个c就是一个轮廓，遍历所有轮廓
    for c in contours:
        # 计算轮廓面积
        if cv2.contourArea(c) > 2500:
            (x, y, w, h) = cv2.boundingRect(c)
            # 绘制矩形（原始图片，左上角起点，右下角终点，线的颜色，线的粗细）
            cv2.rectangle(screen, (x, y), (x + w, y + h), (255, 0, 255), 2)

    cv2.imshow('MOG处理', process)
    cv2.imshow('阈值处理', threshold)
    cv2.imshow('行为检测', screen)
    # 设置键盘指令等待
    key = cv2.waitKey(1) & 0xFF
    # 按'q'键退出循环
    if key == ord('q'):
        break

# 关闭摄像头
camera.release()
# 关闭所有窗口
cv2.destroyAllWindows()
