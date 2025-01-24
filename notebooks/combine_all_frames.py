import os
import shutil

source_dir = '/usb/segment-anything-2-input/'
target_dir = '/usb/segment-anything-2-whole-input/'

for folder in os.listdir(source_dir):
    basename = folder[:36]
    sub_index = int(folder[37:])

    source_folder = os.path.join(source_dir, folder)
    target_folder = os.path.join(target_dir, basename)

    os.makedirs(target_folder, exist_ok=True)

    for file in os.listdir(source_folder):
        index = int(file.split('.')[0])
        source_file = os.path.join(source_folder, file)
        target_file = os.path.join(target_folder, f'{(sub_index*1500+index):05d}.jpg')

        shutil.copyfile(source_file, target_file)
    
