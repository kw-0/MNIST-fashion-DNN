import pandas as pd
# import matplotlib.pyplot as pltp
import numpy as np
import random

raw_train = pd.read_csv("/Users/kylenwilliams/Fashion MNIST/fashion-mnist_train.csv")   # Your path to data file
raw_test = pd.read_csv("/Users/kylenwilliams/Fashion MNIST/fashion-mnist_test.csv")     # Your path to data file

# note to self; don't forget to randomize data
train_data = np.array(raw_train)    # train data array
test_data = np.array(raw_test)      # test data array

E = np.e


class DenseLayer:
    def __init__(self, nrn_inputs, nrn_num):
        self.weights = 0.01 * np.random.randn(nrn_inputs, nrn_num)
        self.biases = np.zeros((1, nrn_num))

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward_pass(self):
        pass


    # def backward_pass(self, inputs, weights):
    #     self.dinputs = inputs
    #     self.dweights = weights


class ReLuActivation:
    def forward_pass(self, inputs):
        self.input = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward_pass(self):
        pass


class SigmoidActivation:
    def forward_pass(self, inputs):
        exp_values = E ** (inputs - np.max(inputs, axis=1, keepdims=True))  # sub by max to keep the vals from exploding
        normalized_exp_vals = exp_values / np.sum(exp_values)
        self.output = normalized_exp_vals
        return self.output

    def backward_pass(self):
        pass


# class Loss:


# just some testing and messing with the class and forward pass:
clothing_idx = random.randint(0, 60000)
print(f"Index of chosen clothing item: {clothing_idx}")

input_layer = DenseLayer(785, 800)
input_layer.forward_pass(train_data[clothing_idx])
actviation_lay1 = SigmoidActivation()

# img = train_data[clothing_idx, 0:784].reshape((28, 28))
# img_r = r[0, 0:784].reshape((28, 28))    # reshaping the flattened array to a 28x28 so plt can show it
# print(r)
# pltp.imshow(img_r, cmap="gray")
# # pltp.imshow(img, cmap="gray")
# pltp.show()
