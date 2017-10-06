import cv2

# 读取第一个摄像头实时画面
camera = cv2.VideoCapture(0)
# 以MOG2格式进行背景分割
mog = cv2.createBackgroundSubtractorMOG2()

# 设置死循环不断获取摄像头读取到的帧
while True:
    # 获取摄像头帧图片数据流
    _, screen = camera.read()
    # 用MOG格式解析获取到的视频数据流
    process = mog.apply(screen)
    # 以窗口的形式显示出来
    cv2.imshow('frame', process)
    # 获取等待键盘指令
    key = cv2.waitKey(1) & 0xFF
    # 按'q'键退出窗口
    if key == ord('q'):
        break

# 关闭摄像头
camera.release()
# 关闭所有窗口
cv2.destroyAllWindows()
