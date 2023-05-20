import os
import random

source_path = r'E:\Python Project\yolov5\data\test\images'
source_list = os.listdir(source_path)
selected_files = random.sample(source_list, 400)

save_path = r'F:\augment_image_test\out\labels'
save_list = selected_files

for i in range(len(selected_files)):
    selected_files[i] = selected_files[i][:-3] + 'txt'
    save_list[i] = os.path.join(save_path,selected_files[i])
    print(save_list[i])
    with open(save_list[i], "w") as empty_file:
        empty_file.write("")
