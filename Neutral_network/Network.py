class Network:
    def __init__(self):
        self.Layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.Layers.append(layer)

    def setup_loss(self,loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input):
        n = len(input)
        result = []
        for i in range(n):
            output = input[i]

            for layer in self.Layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, X_train, y_train, learnin_rate, epochs):
        n = len(X_train)
        for i in range(epochs):
            err = 0
            for j in range(n):
                output = X_train[j]

                for layer in self.Layers:
                    output = layer.forward_propagation(output)

                err += self.loss(y_train[j], output)
                error = self.loss_prime(y_train[j], output)

                for layer in reversed(self.Layers):
                    error = layer.backward_propagation(error, learnin_rate)
            error /= n
            print('epoch: %d/%d err = %f'%(i, epochs, err))