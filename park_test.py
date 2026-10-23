# # - cap = cv2.VideoCapture(video_path) —— 打开视频文件
# # - ret, frame = cap.read() —— 读取一帧， ret 表示是否成功， frame 就是图片
# # - cap.get(cv2.CAP_PROP_FRAME_COUNT) —— 获取视频总帧数
# # - cap.get(cv2.CAP_PROP_FPS) —— 获取帧率（每秒多少帧）
# import cv2
# import os
#
# video_path = r'D:\data\opencv\learn\car-opencv\video\parking_video.mp4'
# frames_dir = r'D:\data\opencv\learn\car-opencv\frames'
#
# # 打开视频
# cap = cv2.VideoCapture(video_path)#打开视频文件或者摄像头
#
# # 获取视频信息
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))#cv2.CAP_PROP_FRAME_COUNT 获取视频总帧数
# fps = cap.get(cv2.CAP_PROP_FPS)#cv2.CAP_PROP_FPS 获取帧率（每秒多少帧）
# print(f"总帧数: {total_frames}")
# print(f"帧率: {fps}")
# print(f"视频时长: {total_frames / fps} 秒")
#
# frame_count = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break# 读取到空帧，说明视频读取完成
#     # 保存当前帧
#     frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
#     # cv2.imwrite(frame_path, frame)
#
#     frame_count += 1
#     frame_interval = 150
#     if frame_count % frame_interval ==0:
#         cv2.imwrite(frame_path, frame)
#         print(f"已保存 {frame_count}/{total_frames} 帧")
#
# cap.release()
# print(f"视频读取完成，共保存 {frame_count} 帧")


#
# import cv2
# import pickle
#
# # 加载车位字典
# with open('spot_dict.pickle','rb') as f:
#     spot_dict = pickle.load(f)
#
# #读取一帧图片
# frame = cv2.imread(r'D:\data\opencv\learn\car-opencv\frames\frame_0449.jpg')
#
# #画车位框
# for spot_coords, spot_id in spot_dict.items():
#     x1,y1,x2,y2 = spot_coords
#     cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
#
# # 显示结果
# cv2.imshow('Parking Frame', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



import cv2
import os
import pickle
import glob

# 加载车位字典
with open('spot_dict.pickle','rb') as f:
    spot_dict = pickle.load(f)

# 文件夹路径
frames_dir = r'D:\data\opencv\learn\car-opencv\frames'
spot_images_dir = r'D:\data\opencv\learn\car-opencv\spot_images'

# 获取所有帧图片
frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))

total_spots = 0
for frame_path in frame_files:
    frame_name = os.path.basename(frame_path).replace('.jpg','')
    # 读取帧图片

    frame = cv2.imread(frame_path)
    if frame is None:
        continue

    # 遍历每一个车位
    for spot_coords,spot_id in spot_dict.items():
        x1,y1,x2,y2 = map(int,spot_coords)
        # 切片截取车位图片
        spo_img = frame[y1:y2,x1:x2]

        # 检查截取是否有效（防止坐标越界）
        if spo_img.size == 0:
            continue

        # 保存：文件名包含帧号和车位号，方便追溯
        spo_filename = f'{frame_name}_spot_{spot_id:03d}.jpg'
        spot_path = os.path.join(spot_images_dir,spo_filename)
        cv2.imwrite(spot_path,spo_img)
        total_spots += 1

print(f"共截取 {total_spots} 个车位")







