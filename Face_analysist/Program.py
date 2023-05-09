from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import numpy as np

import os
SIZE = (100,100)
def getData(n = 1000):
    result = []
    count = 0
    for path in os.listdir("dataset/Image/Face/"):
        if os.path.isfile(os.path.join("dataset/Image/Face/", path)):
            count += 1
            result.append(Image.open(os.path.join("dataset/Image/Face/", path)).convert(mode="L"))
        if count >= n:
            break
    return result

def ChopAndConvert(image: Image.Image):
    data = np.array(image)
    shape = data.shape
    data2 = data[int(shape[0]/4) : int(shape[0]*3/4), :]
    image = Image.fromarray(data2).resize(SIZE)
    return image

def Normalize(array: np.ndarray):
    return array / 255

dataset = getData(1000)
for i in range(len(dataset)):
    # Cắt và chuyển chế độ ảnh
    img = ChopAndConvert(dataset[i])
    # Làm phẳng ảnh
    dataset[i] = np.array(img).reshape((SIZE[0]*SIZE[1], ))

average = sum(np.array(dataset) / len(dataset))
dataset_after_subtract = np.array([arr - average for arr in dataset])
inv = np.linalg.pinv(dataset_after_subtract)
K = 100
C_quote = dataset_after_subtract.dot(dataset_after_subtract.T)

w, v = np.linalg.eig(C_quote)
Keigenvalue = w[0:K]
Keigenvector = v[:, 0:K]
Keigenvector = inv.dot(Keigenvector)
Keigenvector = Keigenvector.astype(dtype='float64')
Keigenvector : np.ndarray

inv2 = np.linalg.pinv(Keigenvector)
W = inv2.dot(dataset_after_subtract[0].reshape(-1,1))
data_result = Keigenvector.dot(W.reshape(-1,1))