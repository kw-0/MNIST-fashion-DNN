import pandas as pd
import matplotlib.pyplot as pltp
import numpy as np

raw_train = pd.read_csv('/Users/kylenwilliams/Fashion MNIST/fashion-mnist_train.csv')   # Your path to data file
raw_test = pd.read_csv('/Users/kylenwilliams/Fashion MNIST/fashion-mnist_test.csv')     # Your path to data file

train_data = np.array(raw_train)    # train data array
test_data = np.array(raw_test)      # test data array

labels = ['T-Shirt', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

E = np.e
EPOCHS = 10
BATCH_SIZE = 4


class DenseLayer:
    def __init__(self, nrn_inputs, nrn_num):
        self.weights = 0.01 * np.random.randn(nrn_inputs, nrn_num)  # Already transposed weights matrix
        self.biases = np.zeros((1, nrn_num))

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward_pass(self):
        pass


    # def backward_pass(self, dinputs):
    #     self.dinputs = inputs
    #     self.dweights = weights


class ReLuActivation:
    def forward_pass(self, inputs):
        self.input = inputs
        self.output = np.maximum(0, inputs)

    def backward_pass(self):
        pass


class SoftmaxActivation:
    def forward_pass(self, inputs):
        exp_values = E ** (inputs - np.max(inputs, axis=1, keepdims=True))  # sub by max to keep the vals from exploding
        normalized_exp_vals = exp_values / np.sum(exp_values)
        self.output = normalized_exp_vals
        return self.output

    def backward_pass(self):
        pass


class CategoricalCrossEntropyLoss:
    def forward(self, pred, truth):
        loss = -np.log(pred, truth)
        self.output = loss


input_layer = DenseLayer(785, 785)
actviation_lay1 = ReLuActivation()

# three hidden layers
layer_2 = DenseLayer(785, 64)
actviation_lay2 = ReLuActivation()
layer_3 = DenseLayer(64, 64)
actviation_lay3 = ReLuActivation()
layer_4 = DenseLayer(64, 64)
actviation_lay4 = ReLuActivation()

output_layer = DenseLayer(64, 9)
output_activation = SoftmaxActivation()


for EPOCH in range(EPOCHS):
    clothing_idx = np.random.randint(0, train_data.shape[0])
    print(f'Epoch {EPOCH + 1}:')

    for BATCH in range(BATCH_SIZE):
        ground_truth = train_data[clothing_idx, 0]   # the first 

        input_layer.forward_pass(train_data[clothing_idx:clothing_idx + BATCH_SIZE])
        actviation_lay1.forward_pass(input_layer.output)

        layer_2.forward_pass(actviation_lay1.output)
        actviation_lay2.forward_pass(layer_2.output)

        layer_3.forward_pass(actviation_lay2.output)
        actviation_lay3.forward_pass(layer_3.output)

        layer_4.forward_pass(actviation_lay3.output)
        actviation_lay4.forward_pass(layer_4.output)

        output_layer.forward_pass(actviation_lay4.output)
        output_activation.forward_pass(output_layer.output)

        print(f'Index of clothing item: {clothing_idx}')

        prediction_idx = np.argmax(output_activation.output)
        prediction = labels[prediction_idx]
        true_label = labels[train_data[clothing_idx, 0]]

        prob_dist = output_activation.output[BATCH]

        print(f'{BATCH+1}- Guess: {prediction}, Truth: {true_label}')
        print(f'Probability distribution across labels: \n {prob_dist}')
        print('')
        clothing_idx += 1

    if EPOCH/5000 == 1:
        img = train_data[clothing_idx, 0:784].reshape((28, 28))  # reshaping the flattened array to so plt can show it
        pltp.imshow(img, cmap='gray')
        pltp.show()
