import cv2
import face_recognition
import numpy as np
import tqdm
import subprocess


def video2mp3(file_name):
    """
    将视频转为音频
    :param file_name: 传入视频文件的路径
    """
    outfile_name = file_name.split('.')[0] + '.mp3'
    cmd = 'ffmpeg -i ' + file_name + ' -f mp3 ' + outfile_name
    # print(cmd)
    subprocess.call(cmd, shell=True)


def video_add_mp3(file_name, mp3_file):
    """
     视频添加音频
    :param file_name: 传入视频文件的路径
    :param mp3_file: 传入音频文件的路径
    """
    outfile_name = file_name.split('.')[0] + '-final.mp4'
    subprocess.call('ffmpeg -i ' + file_name
                    + ' -i ' + mp3_file + ' -strict -2 -f mp4 '
                    + outfile_name, shell=True)


def track_face(frame, tracker, last_box=None):
    """
    在当前帧中追踪目标人脸。如果没有人脸被检测到，继续跟踪上一帧的矩形区域。
    :param frame: 当前帧
    :param tracker: OpenCV跟踪器
    :param last_box: 上一帧的目标区域（如果没有新的目标）
    :return: 跟踪器的更新框
    """
    if tracker is None:
        return None

    if last_box is not None:  # 如果上一帧有目标区域，继续跟踪
        tracker.init(frame, last_box)
        success, box = tracker.update(frame)
        if success:
            return box  # 返回更新后的矩形框
        else:
            return None
    return None


def apply_circle_mask(frame, circle_center, radius, mosaic_img):
    """
    将矩形的马赛克图片裁剪为圆形，并覆盖到视频帧的指定位置。
    :param frame: 视频的当前帧
    :param circle_center: 圆形区域的中心坐标 (x, y)
    :param radius: 圆形区域的半径
    :param mosaic_img: 要应用的马赛克图片
    :return: 带有圆形马赛克的帧
    """
    # 确保半径是整数
    radius = int(radius)

    # 裁剪马赛克图片为圆形
    mask = np.zeros((radius * 2, radius * 2, 3), dtype=np.uint8)
    center = (radius, radius)
    cv2.circle(mask, center, radius, (255, 255, 255), -1)

    # 调整马赛克图片的大小到圆形区域大小
    mosaic_resized = cv2.resize(mosaic_img, (radius * 2, radius * 2))
    circular_mosaic = cv2.bitwise_and(mosaic_resized, mask)

    # 提取圆形区域外的黑色背景掩码
    circular_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, circular_mask = cv2.threshold(circular_mask, 1, 255, cv2.THRESH_BINARY)

    # 定义圆形区域在帧中的位置
    x1, y1 = circle_center[0] - radius, circle_center[1] - radius
    x2, y2 = circle_center[0] + radius, circle_center[1] + radius

    # 确保不越界
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])

    # 获取原始帧中圆形区域的位置
    roi = frame[y1:y2, x1:x2]

    # 调整裁剪后的圆形马赛克和原始帧的 ROI 区域大小一致
    circular_mosaic = cv2.resize(circular_mosaic, (roi.shape[1], roi.shape[0]))
    circular_mask = cv2.resize(circular_mask, (roi.shape[1], roi.shape[0]))

    # 将圆形马赛克叠加到帧的指定位置
    inv_mask = cv2.bitwise_not(circular_mask)
    background = cv2.bitwise_and(roi, roi, mask=inv_mask)
    foreground = cv2.bitwise_and(circular_mosaic, circular_mosaic, mask=circular_mask)
    frame[y1:y2, x1:x2] = cv2.add(background, foreground)

    return frame


def mask_video(input_video, output_video, target_img, mask_img):
    """
    将视频中的特定人脸打码，并添加目标跟踪。
    :param input_video: 要处理的视频
    :param output_video: 处理后的视频
    :param target_img: 要打码的人脸
    :param mask_img: 用于打码的图像
    """
    # 加载目标人脸并获取特征
    trump_image = face_recognition.load_image_file(target_img)
    trump_encoding = face_recognition.face_encodings(trump_image)[0]

    # 加载遮挡用的图片 mask.jpg
    mask_image = cv2.imread(mask_img)

    # 加载 DNN 模型
    modelFile = "deploy.prototxt"  # Caffe 模型配置文件
    weightsFile = "res10_300x300_ssd_iter_140000.caffemodel"  # Caffe 权重文件
    net = cv2.dnn.readNetFromCaffe(modelFile, weightsFile)

    # 打开视频文件
    cap = cv2.VideoCapture(input_video)

    # 获取视频的宽度、高度和帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 计算视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建 VideoWriter 对象，输出视频
    output_filename = output_video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    # 创建一个跟踪器
    tracker = cv2.TrackerCSRT_create()

    # 上一帧的人脸框
    last_box = None

    # 循环读取视频的每一帧，并添加进度条
    with tqdm.tqdm(total=total_frames, desc="Processing Frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()  # 读取一帧
            if not ret:
                break  # 如果没有读取到帧，退出循环

            # 将帧转换为输入格式
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)

            # 进行人脸检测
            detections = net.forward()

            face_detected = False  # 标志，表示是否检测到人脸

            # 在检测到的人脸周围绘制框，并进行相似度比较
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.25:  # 置信度阈值
                    box = detections[0, 0, i, 3:7] * np.array(
                        [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (left, top, right, bottom) = box.astype("int")

                    # 确保坐标合法
                    left = max(0, left)
                    top = max(0, top)
                    right = min(frame_width, right)
                    bottom = min(frame_height, bottom)

                    # 提取人脸区域并计算相似度
                    face_image = frame[top:bottom, left:right]
                    if face_image.size > 0:  # 检查是否为非空区域
                        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                        face_encoding = face_recognition.face_encodings(face_image_rgb)

                        if face_encoding:
                            face_encoding = face_encoding[0]
                            matches = face_recognition.compare_faces([trump_encoding], face_encoding)
                            face_distances = face_recognition.face_distance([trump_encoding], face_encoding)
                            similarity = 1 - face_distances[0]  # 相似度越高，距离越小

                            if matches[0] and similarity > 0.25:
                                # 初始化跟踪器
                                last_box = (left, top, right - left, bottom - top)
                                tracker.init(frame, last_box)
                                face_detected = True
                                break
            if not face_detected and last_box is not None:
                # 如果没有检测到新的人脸，使用跟踪器跟踪上一帧的区域
                last_box = track_face(frame, tracker, last_box)

            if last_box is not None:
                left, top, w, h = last_box
                # 计算圆心和半径
                circle_center = (int(left + w / 2), int(top - 15 + h / 2))
                radius = int(np.sqrt(w ** 2 + h ** 2) / 2)
                # 绘制圆形框
                frame = apply_circle_mask(frame, circle_center, radius, mask_image)

            # 将处理后的帧写入输出视频
            out.write(frame)

            # 更新进度条
            pbar.update(1)

    # 释放视频捕捉对象和输出对象
    cap.release()
    out.release()

    print(f"输出视频已保存为: {output_filename}")


def process_video(video_path, face_path, mosaic_path, output_path):
    # 保存音频为mp3
    video2mp3(file_name=video_path)
    # 处理视频，自动打码，输出视频为output.mp4
    mask_video(input_video=video_path, output_video=output_path, target_img=face_path, mask_img=mosaic_path)
    # 为 output.mp4 处理好的视频添加声音
    video_add_mp3(file_name=output_path, mp3_file=video_path.split('.')[0] + '.mp3', )


if __name__ == '__main__':
    # 要处理的视频
    target_video = 'short.mp4'
    # 要替换的人脸
    target_face = 'target_face.jpg'
    # 替换用的图像
    mosaic_img = 'mask.jpg'
    # 打码后的视频名称
    output_name = 'masked_output.mp4'
    # 处理视频
    process_video(target_video, target_face, mosaic_img, output_name)
