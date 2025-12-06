#Khai báo thư viện
import os
import cv2
import numpy as np
#Đọc và xử lý dữ liệu từ File Dataset
def Read_Dataset(path, size= (64,64)):
    Pictures= []
    for file in os.listdir(path):
        file_path= os.path.join(path, file)
        image= cv2.imread(file_path)
        image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image= cv2.resize(image,size)
        Pictures.append(image.flatten())
    Pictures = np.array(Pictures).astype(np.float32) / 255.0
    return Pictures


