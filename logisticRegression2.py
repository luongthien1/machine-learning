import numpy as np
from function import readnext
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

if __name__ == "__main__":
    data = getdata()

    x = np.concatenate((data[:, 0:-1], np.ones((data.shape[0],1))), axis=1)
    y = data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
    lorg = LogisticRegression()
    lorg.fit(X_train, y_train)
    print(lorg.score(X_test, y_test))
