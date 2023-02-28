import os

'''按标签修改训练集名称'''

path = r'D:/桌面/学习/毕业设计/数据集/图片/2/Train_Alphabet/'
folders = os.listdir(path) # 获取所有文件夹名的列表

for folder in folders:
    folder_name =path + folder # 原文件夹完整路径
    files = os.listdir(folder_name) # 获取新文件夹下所有文件名的列表

    num = 0
    for file in files:
        old_file_name = folder_name + '/' + file # 原文件完整路径
        new_file_name = folder_name + str(num) + '.png'# 新文件完整路径

        os.rename(old_file_name, new_file_name) # 修改文件名
        num = num + 1