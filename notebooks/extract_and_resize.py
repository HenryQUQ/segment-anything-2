import cv2
import os
from tqdm import tqdm

root ='/bask/projects/j/jiaoj-3d-vision/360x_official_launch/360x_dataset/360x_dataset_original_resolution/panoramic'

video_list = os.listdir(root)

video_list = [os.path.join(root, path) for path in video_list]
# for video_folder in video_folders:
#     video_folder_path = os.path.join(root, video_folder)
#     if os.path.isdir(video_folder_path):
#         video_path = os.path.join(video_folder_path, '360', '360_panoramic.mp4')
#         if os.path.exists(video_path):
#             video_list.append(video_path)

output_folder = '/bask/projects/j/jiaoj-3d-vision/360XProject/segment-anything-2-input'
os.makedirs(output_folder, exist_ok=True)



def extract_and_resize_frames(video_path, output_folder, new_size=(1080, 540)):

    basename = os.path.basename(video_path)[:-4]
    cap = cv2.VideoCapture(video_path)

    # max_frame_number = 1500


    frame_number = 0

    sub_folder = os.path.join(output_folder, f"{basename}")
    os.makedirs(sub_folder, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        resized_frame = cv2.resize(frame, new_size)

        # sub_folder_index = frame_number // max_frame_number
        #
        # index_in_sub_folder = frame_number % max_frame_number





        # 构造输出文件路径
        output_path = os.path.join(sub_folder, f'{frame_number:05d}.jpg')

        # 保存调整大小后的帧
        cv2.imwrite(output_path, resized_frame)

        frame_number += 1



    # 释放视频捕获对象
    cap.release()
    print(f"所有帧已提取并保存到 {output_folder}")


for video_path in tqdm(video_list):
    extract_and_resize_frames(video_path, output_folder)

