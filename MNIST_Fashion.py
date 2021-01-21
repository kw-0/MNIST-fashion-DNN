import pandas as pd
import matplotlib.pyplot as pltp
import numpy as np
import random

train_d = pd.read_csv("/Users/kylenwilliams/Fashion MNIST/fashion-mnist_train.csv")   # Your path to data file
test_d = pd.read_csv("/Users/kylenwilliams/Fashion MNIST/fashion-mnist_test.csv")     # Your path to data file

# note to self; don't forget to randomize and batch the data
train_data = np.array(train_d)    # train data array
test_data = np.array(test_d)      # test data array


class DenseLayer:
    def __init__(self, nrn_inputs, nrn_num):
        self.weights = 0.01 * random.randint(len(nrn_inputs), nrn_num)
        self.biases = np.zeros((1, len(nrn_inputs)))

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward_pass(self, inputs, weights):
        self.dinputs = inputs
        self.dweights = weights


# just some testing and messing with the class and forward pass:
clothing_idx = random.randint(0, 60000)
print(f"Index of chosen clothing item: {clothing_idx}")

input_layer = DenseLayer(train_data[clothing_idx], 785)
input_layer_out = input_layer.forward_pass(train_data[clothing_idx])
# print(input_layer_out)  # layer's output

img = train_data[clothing_idx, 0:784].reshape((28, 28))    # reshaping the flattened array to a 28x28 so plt can show it
pltp.imshow(img, cmap="gray")
pltp.show()
