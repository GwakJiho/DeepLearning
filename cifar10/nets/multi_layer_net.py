import numpy as np
from collections import OrderedDict

from utils.optimizer import Adam
from nets.layers import Affine, Sigmoid, Relu, LeakyReLU,  SoftmaxWithLoss, Dropout, BatchNormalization


class Model:

    def __init__(self, input_size, hidden_size_list, output_size, weight_init_std=0.01,
                 use_batchnorm=False, weight_decay_lambda=0, use_dropout=False, dropout_ratio=0.2,
                 use_weights=False, weights=""):
        self.params = {}
        self.use_batchnorm = use_batchnorm
        self.weight_decay_lambda = weight_decay_lambda
        self.use_dropout = use_dropout
        
        if use_weights == True:  #가중치 업데이트
            self.params = np.load(weights, allow_pickle=True).tolist()
        else:
            self.params = {}
            

            self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size_list[0])
            self.params['b1'] = np.zeros(hidden_size_list[0])
            self.params['W2'] = weight_init_std * np.random.randn(hidden_size_list[0], hidden_size_list[1])
            self.params['b2'] = np.zeros(hidden_size_list[1])
            self.params['W3'] = weight_init_std * np.random.randn(hidden_size_list[1], hidden_size_list[2])
            self.params['b3'] = np.zeros(hidden_size_list[2])
            self.params['W4'] = weight_init_std * np.random.randn(hidden_size_list[2], output_size)
            self.params['b4'] = np.zeros(output_size)

            if self.use_batchnorm:
                self.params['gamma1'] = np.ones(hidden_size)
                self.params['beta1'] = np.zeros(hidden_size)

        

        self.layers = OrderedDict()

        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])

        # Batch Normalization Layer  batch normar 은 사용 안할 듯
        if self.use_batchnorm:
            self.layers['BatchNorm1'] = BatchNormalization(self.params['gamma1'], self.params['beta1'])
        if self.use_dropout:  #드랍아웃
            self.layers['Dropout1'] = Dropout(dropout_ratio)

        self.layers['Relu1'] = LeakyReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        if self.use_dropout:  
            self.layers['Dropout2'] = Dropout(dropout_ratio)
        self.layers['Relu2'] = LeakyReLU()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        if self.use_dropout: 
            self.layers['Dropout3'] = Dropout(dropout_ratio)
        self.layers['sigmoid1'] = Sigmoid()
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])

        self.lastLayer = SoftmaxWithLoss()

        
    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
      
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
       
        return accuracy

    def numerical_gradient(self, f, x):
        h = 1e-4  # 0.0001
        grad = np.zeros_like(x)

        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(x)  # f(x+h)

            x[idx] = tmp_val - h
            fxh2 = f(x)  # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2 * h)

            x[idx] = tmp_val  # 값 복원
            it.iternext()

        return grad

    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)  

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}

        # For Weight Decay

        grads['W1'] = self.layers['Affine1'].dW 
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW 
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW 
        grads['b3'] = self.layers['Affine3'].db
        grads['W4'] = self.layers['Affine4'].dW 
        grads['b4'] = self.layers['Affine4'].db

        if self.use_batchnorm:
            grads['gamma1'] = self.layers['BatchNorm1'].dgamma
            grads['beta1'] = self.layers['BatchNorm1'].dbeta

        return grads  # dictionary 형

