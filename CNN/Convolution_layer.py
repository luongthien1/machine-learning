import numpy as np

class ConvolutionLayer:

    def __init__(self, K: np.ndarray, Padding: int = 1, Stride: int = 1):
        self.K = K
        self.Padding = Padding
        self.Stride = Stride
        self.result = None
    
    def input(self, data: np.ndarray):
        self.data = data

    def process(self):
        result = []
        klength = len(self.K)

        padding_data = np.zeros((self.data.shape[0] + 2*self.Padding, self.data.shape[1] + 2*self.Padding, self.data.shape[2]))
        padding_data[1:-1, 1:-1, :] = self.data

        for ik in range(klength):
            result.append(self.calculate(self.K[ik], padding_data))
        self.result = np.array(result)

    def calculate(self, kernel: np.ndarray, data: np.ndarray):
        width,height,deep = self.data.shape
        kwidth, kheight, kdeep = kernel.shape
        result = np.zeros((int((width - kwidth + 2*self.Padding)/self.Stride + 1), int((height - kheight + 2*self.Padding)/self.Stride + 1), deep))
        for iw in range(result.shape[0]):
            for ih in range(result.shape[1]):
                s = 0
                for id in range(deep):
                    img = data[iw*self.Stride: iw*self.Stride + kwidth, ih*self.Stride: ih*self.Stride + kheight, id]
                    ker = kernel[:, :, id]
                    s += np.sum(img * kernel[:, :, id])
                result[iw, ih, :] = s/3, s/3, s/3
        return result
    