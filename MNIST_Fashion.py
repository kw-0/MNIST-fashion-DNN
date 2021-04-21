import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# could implement several different NNs (each with various tweaked values) that could vote to increase accuracy
# possibly use os instead next time
raw_train = pd.read_csv('/Users/kylenwilliams/Fashion MNIST/fashion-mnist_train.csv')
raw_test = pd.read_csv('/Users/kylenwilliams/Fashion MNIST/fashion-mnist_test.csv')

train_data = np.array(raw_train)  # train data array
test_data = np.array(raw_test)  # test data array

# data preprocessing
np.random.shuffle(train_data)
np.random.shuffle(test_data)

EPOCHS = 1
BATCH_SIZE = 2

labels = ['T-Shirt', 'Trousers', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']


def get_truths(inputs):
    # could be optimized
    # puts all the ground truths into their own array
    ground_truths = []
    for data_point in inputs:
        ground_truths = np.append(ground_truths, [data_point[0]], 0)  # adds truth values to ground true array
        # print(np.shape(train_data)[0]//BATCH_SIZE, [data_point[0]], 0)
    ground_truths = ground_truths.astype(int)

    ground_truths = ground_truths.reshape(-1)  # flattens the ground truths array
    inputs = np.delete(inputs, 0, axis=1)  # deletes all the truths from input data
    return ground_truths, inputs


class DenseLayer:
    # inits a layer and defines the regularizers
    def __init__(self, nrn_inputs, nrn_num, l1w_regularizer=0, l1b_regularizer=0, l2w_regularizer=0, l2b_regularizer=0):
        self.weights = 0.01 * np.random.randn(nrn_inputs, nrn_num)  # already transposed weights matrix
        self.biases = np.zeros((1, nrn_num))
        self.l1w_regularizer = l1w_regularizer
        self.l1b_regularizer = l1b_regularizer
        self.l2w_regularizer = l2w_regularizer
        self.l2b_regularizer = l2b_regularizer

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward_pass(self, dvalues):
        """
        dinputs is initiliaized like this because the partial derivative of each input is the respective weight,
        and we want to use the chain rule to multiply by the dvalues of the previous neurons, so to get one
        value for each neuron derivative, we add the result of each inputs/weights derivative in each neuron so
        we have one dvalue to pass back to the previous layer. Also the dvalues * inputs (or weights) are added
        per neuron over all batches so we can actually get the benefits of batching
        """
        # weights derivative first so it takes the shape of weights
        self.dweights = np.dot(self.inputs.T, dvalues)

        # bias is added del of addition is 1
        # so all we have to do is the adding of the dvalues for each neuron
        # over all the batches
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # L1 on weights
        if self.l1w_regularizer > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.l1w_regularizer * dL1
        # L2 on weights
        elif self.l2w_regularizer > 0:
            self.dweights += 2 * self.l2w_regularizer * \
                             self.weights
        # L1 on biases
        if self.l1b_regularizer > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.l1b_regularizer * dL1
        # L2 on biases
        elif self.l2b_regularizer > 0:
            self.dbiases += 2 * self.l2b_regularizer * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)


class Dropout:
    def __init__(self, dropout_rate):
        # 1- bc we want the 1- drop rate to stay behind
        self.dropout_rate = 1 - dropout_rate

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.binomial_dist = np.random.binomial(1, 1 - self.dropout_rate, size=inputs.shape)

        self.output = inputs * self.binomial_dist

    def backard_pass(self, dvalues):
        # gradient calc
        self.dinputs = dvalues * self.binomial_dist


# maybe use leaky ReLu instead
class ReLuActivation:
    def forward_pass(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward_pass(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class SoftmaxActivation:
    def forward_pass(self, inputs):
        # sub by max to keep the values from exploding
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        normalized_exp_vals = exp_values / np.sum(exp_values)
        self.output = normalized_exp_vals
        return self.output  # without the return I had some NoneType errors so this fixed that

    # we don't actually use this backward pass but its nice to have (we use the combined one)
    # also btw idx means index future me
    def backward_pass(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        # enumerate outputs and gradients
        for idx, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # flatten output array
            single_output = single_output.flatten

            # this is just how the derivative math worked out (and was then optimized in code form)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[idx] = np.dot(jacobian_matrix, single_dvalues)

    def get_params(self):
        return self.output


class GeneralLoss:
    def calculate(self, output, y):
        # this works b/c its only called in the inherited categorical cross entropy loss class
        sample_losses = self.forward_pass(output, y)

        # calculate mean loss
        mean_loss = np.mean(sample_losses)

        return mean_loss

    def regularization(self, layer):
        regularization_loss = 0

        # checks if you input l1 regularizer (lambda) for weights into the call of the layer
        if layer.l1w_regularizer > 0:
            l1w = layer.l1w_regularizer * sum(abs(layer.weights))
            regularization_loss += l1w

        # checks if you input l1 regularizer (lambda) for biases into the call of the layer
        if layer.l1b_regularizer > 0:
            l1b = layer.l1b_regularizer * sum(abs(layer.weights))
            regularization_loss += l1b

        # checks if you input l2 regularizer (lambda) weights into the call of the layer
        if layer.l2w_regularizer > 0:
            l2w = layer.l2w_regularizer * sum(abs(layer.weights))
            regularization_loss += l2w

        # checks if you input l2 regularizer (lambda) biases into the call of the layer
        if layer.l2b_regularizer > 0:
            l2b = layer.l2b_regularizer * sum(abs(layer.weights))
            regularization_loss += l2b

        return regularization_loss


class CategoricalCrossEntropyLoss(GeneralLoss):
    def forward_pass(self, y_pred, y_true):
        correct_confidences = 0
        samples = len(y_pred)

        # Clip data to prevent log(0)
        # Clip both sides to not drag to anything
        y_pred_clipped = np.clip(y_pred, 1e-7, (1 - 1e-7))

        # the data here is sparse but I want that delicious scalability so one-hot is included
        if len(y_true.shape) == 1:  # if sparse
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true[range(samples)]
            ]

        elif len(y_true.shape) == 2:  # if one-hot
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # we don't actually use this backward pass but its nice to have (we use the combined one)
    def backward_pass(self, dvalues, y_true):
        # sample amount
        samples = len(dvalues)

        # turns sparse labels into one hot
        if len(y_true.shape) == 1:
            y_true = np.eye(len(labels))[y_true]

        # Calculate gradient then normalize it
        self.dinputs = (-y_true / dvalues)
        self.dinputs = self.dinputs / samples


# combined for faster back prop
class ActivationSoftmaxLossCategoricalCrossEntropy:
    def __init__(self):
        self.activation = SoftmaxActivation()
        self.loss = CategoricalCrossEntropyLoss()

    def forward_pass(self, inputs, y_true):
        self.output = self.activation.forward_pass(inputs)

        return self.loss.calculate(self.output, y_true)

    def backward_pass(self, dvalues, y_true):
        samples = len(dvalues)

        self.dinputs = dvalues.copy()

        # one hot --> sparse
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs[range(samples), y_true] -= 1  # the derivative of both combined is simplified down to subtracting 1
        self.dinputs = self.dinputs / samples

    def regularization(self, layer):    # alternatively you could do softmax_activation_loss.loss.regularization below
        return self.loss.regularization(layer)


class Optimizer_SGD:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    # Update parameters
    def update_params(self, layer):
        layer.weights += (-self.learning_rate * layer.dweights)
        layer.biases += (-self.learning_rate * layer.dbiases)


input_layer = DenseLayer(784, 784)
activation_lay1 = ReLuActivation()

layer_2 = DenseLayer(784, 64, l2w_regularizer=5e-4, l2b_regularizer=5e-4)
activation_lay2 = ReLuActivation()
layer_3 = DenseLayer(64, 64, l2w_regularizer=5e-4, l2b_regularizer=5e-4)
activation_lay3 = ReLuActivation()
layer_4 = DenseLayer(64, 64, l2w_regularizer=5e-4, l2b_regularizer=5e-4)
activation_lay4 = ReLuActivation()

# layer_5 = DenseLayer(533, 533)
# activation_lay5 = ReLuActivation()
# layer_6 = DenseLayer(533, 533)
# activation_lay6 = ReLuActivation()
# layer_7 = DenseLayer(533, 533)
# activation_lay7 = ReLuActivation()
# layer_8 = DenseLayer(533, 533)
# activation_lay8 = ReLuActivation()
# layer_9 = DenseLayer(533, 533)
# activation_lay9 = ReLuActivation()

output_layer = DenseLayer(64, 10)
softmax_activation_loss = ActivationSoftmaxLossCategoricalCrossEntropy()

dropout1 = Dropout(0.3)
dropout2 = Dropout(0.3)
dropout3 = Dropout(0.3)

optimizer = Optimizer_SGD(0.007)

ground_truths, train_data = get_truths(train_data)

train_data = (train_data.astype(np.float32) - 127.5) / 127.5
test_data = (test_data.astype(np.float32) - 127.5) / 127.5

g = []
for epoch in range(EPOCHS):
    batch_data = train_data[epoch * BATCH_SIZE: (1 + epoch) * BATCH_SIZE]
    batch_truths = ground_truths[epoch * BATCH_SIZE: (1 + epoch) * BATCH_SIZE]

    # forward pass
    input_layer.forward_pass(batch_data)
    activation_lay1.forward_pass(input_layer.output)

    layer_2.forward_pass(activation_lay1.output)
    activation_lay2.forward_pass(layer_2.output)

    dropout1.forward_pass(activation_lay2.output)

    layer_3.forward_pass(dropout1.output)
    activation_lay3.forward_pass(layer_3.output)

    dropout2.forward_pass(activation_lay3.output)

    layer_4.forward_pass(activation_lay3.output)
    activation_lay4.forward_pass(layer_4.output)

    dropout3.forward_pass(activation_lay4.output)

    # layer_5.forward_pass(activation_lay4.output)
    # activation_lay5.forward_pass(layer_5.output)

    # layer_6.forward_pass(activation_lay5.output)
    # activation_lay6.forward_pass(layer_6.output)

    # layer_7.forward_pass(activation_lay6.output)
    # activation_lay7.forward_pass(layer_7.output)

    # layer_8.forward_pass(activation_lay7.output)
    # activation_lay8.forward_pass(layer_8.output)

    # layer_9.forward_pass(activation_lay8.output)
    # activation_lay9.forward_pass(layer_9.output)

    output_layer.forward_pass(dropout3.output)
    softmax_activation_loss.forward_pass(output_layer.output, batch_truths)

    # for i in range(BATCH_SIZE):
    softmax_activation_loss.forward_pass(output_layer.output, batch_truths)

    loss = np.sum(softmax_activation_loss.forward_pass(output_layer.output, batch_truths) +
           softmax_activation_loss.regularization(layer_2) +
           softmax_activation_loss.regularization(layer_3) +
           softmax_activation_loss.regularization(layer_4))

    print(epoch)
    # for batch in range(BATCH_SIZE):
    #     # finds which the neural net predicted
    #     # print(np.max(softmax_activation_loss.output[batch]))
    #     # print(softmax_activation_loss.output[batch])
    prediction = np.argmax(softmax_activation_loss.output)
    #     prediction_label = labels[prediction]
    #
    #     print(f'Batch {batch + 1}- guess: {prediction_label} truth: {labels[ground_truths[count]]}')
    #     prob_dist = softmax_activation_loss.output
    print(f'loss: {loss}')

    # accuracy calc
    if len(ground_truths.shape) == 2:
        ground_truths = np.argmax(ground_truths, axis=1)
    accuracy = np.mean(prediction == ground_truths) * 100

    # backward pass is after loss and stuff because it needs to change for next layer
    softmax_activation_loss.backward_pass(softmax_activation_loss.output, batch_truths)
    output_layer.backward_pass(softmax_activation_loss.dinputs)

    # activation_lay9.backward_pass(output_layer.dinputs)
    # layer_9.backward_pass(activation_lay9.dinputs)
    #
    # activation_lay8.backward_pass(layer_9.dinputs)
    # layer_8.backward_pass(activation_lay8.dinputs)
    #
    # activation_lay7.backward_pass(layer_8.dinputs)
    # layer_7.backward_pass(activation_lay7.dinputs)
    #
    # activation_lay6.backward_pass(layer_7.dinputs)
    # layer_6.backward_pass(activation_lay6.dinputs)
    #
    # activation_lay5.backward_pass(layer_6.dinputs)
    # layer_5.backward_pass(activation_lay5.dinputs)

    dropout3.backard_pass(output_layer.dinputs)

    activation_lay4.backward_pass(output_layer.dinputs)
    layer_4.backward_pass(activation_lay4.dinputs)

    dropout2.backard_pass(layer_4.dinputs)

    activation_lay3.backward_pass(dropout3.dinputs)
    layer_3.backward_pass(activation_lay3.dinputs)

    dropout1.backard_pass(layer_3.dinputs)

    activation_lay2.backward_pass(layer_3.dinputs)
    layer_2.backward_pass(activation_lay2.dinputs)

    activation_lay1.backward_pass(layer_2.dinputs)
    input_layer.backward_pass(activation_lay1.dinputs)

    optimizer.update_params(input_layer)
    optimizer.update_params(layer_2)
    optimizer.update_params(layer_3)
    optimizer.update_params(layer_4)
    # optimizer.update_params(layer_5)
    # optimizer.update_params(layer_6)
    # optimizer.update_params(layer_7)
    # optimizer.update_params(layer_8)
    # optimizer.update_params(layer_9)
    optimizer.update_params(output_layer)

    print(f'Accuracy: {float(accuracy)}%\n')

    # # if you're on the last epoch show that img
    # if EPOCH+1/EPOCHS == 1:
    #     img = train_data[clothing_idx, 1:784].reshape((28, 28))  # reshaping the flattened array to so plt can show it
    #     pltp.imshow(img, cmap='gray')
    #     pltp.show()
