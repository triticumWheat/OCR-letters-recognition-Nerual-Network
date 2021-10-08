import numpy as np

class Data:
    def __init__(self, name, batch_size):
        with open(name, 'rb') as f:
            data = np.load(f, allow_pickle=True)
        self.x = data[0]
        self.y = data[1]
        self.l = len(self.x)
        self.batch_size = batch_size
        self.pos = 0

    def forward(self):
        pos = self.pos
        bat = self.batch_size
        l = self.l
        if pos + bat >= l:
            ret = (self.x[pos:l], self.y[pos:l])
            self.pos = 0
            index = range(l)
            np.random.shuffle(list(index))
            self.x = self.x[index]
            self.y = self.y[index]
        else:
            ret = (self.x[pos:pos + bat], self.y[pos:pos + bat])
            self.pos += self.batch_size

        return ret, self.pos

    def backward(self, d):
        pass


class FullyConnected:
    def __init__(self, l_x, l_y, learningRate):
        self.weight = np.random.randn(l_y, l_x) / np.sqrt(l_x)
        self.bias = np.random.randn(l_y, 1)
        self.learningRate = learningRate

    def forward(self, input):
        self.input = input
        output = np.array([np.dot(self.weight, xx) + self.bias for xx in input])
        return output

    def backward(self, d):
        ddw = [np.dot(dd, input.T) for input, dd in zip(self.input, d)]
        dweight = np.sum(ddw, axis=0) / d.shape[0]
        dbias = np.sum(d, axis=0) / d.shape[0]
        dx = np.array([np.dot(self.weight.T, dd) for dd in d])
        self.weight = self.weight - self.learningRate * dweight
        self.bias = self.bias - self.learningRate * dbias
        return dx


class Sigmoid:
    def __init__(self):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.x = x
        self.y = self.sigmoid(x)
        return self.y

    def backward(self, d):
        sig = self.sigmoid(self.x)
        self.dx = d * sig * (1 - sig)
        return self.dx


class QuadraticLoss:
    def __init__(self):
        pass

    def forward(self, x, label):
        self.x = x
        self.label = np.zeros_like(x)
        for a, b in zip(self.label, label):
            a[b] = 1.0
        loss = np.sum(np.square(self.x - self.label)) / self.x.shape[0] / 2
        return loss

    def backward(self):
        self.dx = (self.x - self.label) / self.x.shape[0]
        return self.dx


class Accuracy:
    def __init__(self):
        pass

    def forward(self, x, label):
        self.accuracy = np.sum([np.argmax(xx) == ll for xx, ll in zip(x, label)])
        self.accuracy = self.accuracy * 1.0 / x.shape[0]
        return self.accuracy

def main():
    datalyer1 = Data("C:\\Users\\13794\\Desktop\\python\\train.npy", 1024)
    datalyer2 = Data("C:\\Users\\13794\\Desktop\\python\\validate.npy", 10000)
    innerlyer = []
    innerlyer.append(FullyConnected(17 * 17, 42, 3000))
    innerlyer.append(Sigmoid())
    innerlyer.append(FullyConnected(42, 42, 3000))
    innerlyer.append(Sigmoid())
    innerlyer.append(FullyConnected(42, 26, 3000))
    innerlyer.append(Sigmoid())
    losslyer = QuadraticLoss()
    accu = Accuracy()

    epoch = 150
    for i in range(epoch):
        print("epoch: ", i)
        lossSum = 0
        iters = 0
        while True:
            data, pos = datalyer1.forward()
            x, label = data
            for layer in innerlyer:
                x = layer.forward(x)
            loss = losslyer.forward(x, label)
            lossSum += loss
            iters += 1
            d = losslyer.backward()
            for layer in innerlyer[::-1]:
                d = layer.backward(d)
            if pos == 0:
                data, _ = datalyer2.forward()
                x, label = data
                for layer in innerlyer:
                    x = layer.forward(x)
                accur = accu.forward(x, label)
                print("accuracy: ", accur * 100, "%")
                print("loss: ", lossSum / iters)
                break

if __name__ == '__main__':
    main()
