import cv2

# 读取摄像头信息
camera = cv2.VideoCapture(0)

# 设置膨胀结构
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
# 初始化背景为None
background = None

while True:
    # 读取视频流
    _, screen = camera.read()
    # 对帧进行预处理，先转灰度图，再进行高斯滤波。
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    '''
    用高斯滤波进行模糊处理，进行处理的原因：
    每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。
    对噪声进行平滑是为了避免在运动和跟踪时将其检测出来
    '''
    screen_gray = cv2.GaussianBlur(screen_gray, (21, 21), 0)

    # 将第一帧设置为整个输入的背景
    if background is None:
        background = screen_gray
        continue
    # 对于每个从背景之后读取的帧都会计算其与背景（前一帧）之间的差异，并得到一个差分图（different map）
    '''
    还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，
    从而对孔（hole）和缺陷（imperfection）进行归一化处理
    '''
    # 获取当前帧与前一帧（背景）之间的分差图
    diff = cv2.absdiff(background, screen_gray)
    # 进行二值化阈值处理
    _, diff = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY)
    # 形态学膨胀，为了是识别效果更好，主要是边缘的膨胀使识别对象圆滑
    diff = cv2.dilate(diff, es, iterations=2)

    # 计算轮廓坐标
    _, contours, _ = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    # 显示每一个识别轮廓
    for c in contours:
        # 对于小于2000的面积变化，不给予显示
        if cv2.contourArea(c) < 2000:
            continue
        # 获取矩形的边界框坐标数据
        (x, y, w, h) = cv2.boundingRect(c)
        # 绘制边界
        cv2.rectangle(screen, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 窗口显示
    cv2.imshow('原图', screen)
    cv2.imshow('处理后', diff)

    # 设置键盘指令等待
    key = cv2.waitKey(1) & 0xFF
    # 按'q'键退出循环
    if key == ord('q'):
        break

# 关闭摄像头
camera.release()
# 关闭所有窗口
cv2.destroyAllWindows()
