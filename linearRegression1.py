import numpy as np
import pandas as pd
from function import readnext

def getdata():
    data = []
    f = open("./dataset/LinearRegression")
    char = readnext(f)
    while (char != "Ket"):
        x = float(char)
        char = readnext(f)
        char = readnext(f)
        y = float(char)
        char = readnext(f)
        data.append([x,y])
    return np.array(data)

def d_loss(x, y_pred, y):
    h = y_pred - y
    result = x*(y_pred - y)
    return x*(y_pred - y)
def f_loss(y_pred, y):
    return (y_pred-y)**2

def linearRegression(w_init : np.ndarray, y : np.ndarray, x : np.ndarray, learning_rate = 0.01, stop = 1000):
    w = [w_init]
    for j in range(stop):
        indexs = np.random.permutation(len(data))
        loss = 0
        w1 = w[-1]
        for i in indexs:
            y_pred = x[i].dot(w[-1])
            w_new = w[-1] - d_loss(x[i], y_pred, y[i])*0.1*learning_rate
            w.append(w_new)        
            loss += f_loss(y_pred, y[i])
        
    return w

if __name__ == "__main__":
    data = getdata()
    x = np.concatenate((data[:, 0:-1], np.ones((data.shape[0],1))), axis=1)
    y = data[:, -1]
    w = linearRegression(np.array([0,0]), y, x, 0.01,10000)
    print(w[-1])