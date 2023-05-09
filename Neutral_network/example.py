from Network import Network
from Layer import Layer
from FullConnectedLayer import FullConnectedLayer
from Activation_layer import ActivationLayer
import numpy as np

#hàm relu
def relu(z: np.ndarray):
    return np.where(z<0,0,z)

# đạo hàm relu
def relu_prime(z : np.ndarray):
    return np.where(z<0,0,1)

# hàm loss
def loss(y_true, y_predict):
    return 0.5*(y_predict-y_true)**2

# đạo hàm hàm loss
def loss_prime(y_true, y_predict):
    return y_predict - y_true

# tập train 2D
X_train = np.array((((0,0),), ((0,1),), ((1,0),), ((1,1),)))

y_train = np.array((((0),), ((1),), ((1),), ((0),)))

net = Network()
net.add(FullConnectedLayer((1,2), (1,3))) # Full Connected Layer 1
net.add(ActivationLayer(relu, relu_prime))  # lớp 1 nối với lớp kích hoạt relu
net.add(FullConnectedLayer((1,3), (1,1))) # Full Connected Layer 2
net.add(ActivationLayer(relu, relu_prime))  # lớp 2 nối với lớp kích hoạt relu

# setup hàm loss
net.setup_loss(loss, loss_prime)
net.fit(X_train, y_train, 0.01, 1000)

out = net.predict(np.array(((0,1),)))

# vì số lượng train thấp, cần chạy nhiều lần để đạt kết quả chính xác.
print(out)