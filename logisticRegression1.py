import numpy as np
from function import readnext
import matplotlib.pyplot as plt

def getdata():
    data = []
    f = open("./dataset/logisticRegression")
    char = readnext(f)
    while (char != "ket"):
        x = float(char)
        char = readnext(f)
        char = readnext(f)
        y = float(char)
        char = readnext(f)
        data.append([x,y])
    return np.array(data)

def d_loss(x, y_pred, y):
    h = y_pred - y
    result = np.array([x[i] * h[i] for i in range(len(x))])
    return np.sum(result, axis=0)/x.shape[0]
def f_loss(y_pred, y):
    b =y*np.log(np.where(y_pred == 0, 0.00001, y_pred)) 
    h = 1-y_pred
    h = np.where(h == 0, 0.00001, h)
    a = (1-y)*np.log(h)
    return np.sum(-(a+b))/y.shape[0]

def sigmoid(w,x):
    return 1/(1+np.exp(-x.dot(w)))

def logisticRegression(w_init : np.ndarray, y : np.ndarray, x : np.ndarray, learning_rate = 0.01, stop = 0.0001):
    w = [w_init]
    while True:
        indexs = np.random.permutation(len(data))
        loss = 0
        w1 = w[-1]
        y_pred = sigmoid(w[-1], x)

        w_new = w[-1] - d_loss(x, y_pred, y)*learning_rate
        w.append(w_new)        
        loss += f_loss(y_pred, y)
        if loss < stop:
            break
    return w, loss

if __name__ == "__main__":
    data = getdata()
    x = np.concatenate((data[:, 0:-1], np.ones((data.shape[0],1))), axis=1)
    y = data[:, -1]
    w = logisticRegression(np.array([0,0]), y, x, 2,10000)
    print(w[-1])

    plt.scatter(x[:,0],y, marker="o")
    X = np.linspace([-50,1],[50,1],1000)
    plt.plot(X[:,0], sigmoid(w[-1],X))
    plt.show()