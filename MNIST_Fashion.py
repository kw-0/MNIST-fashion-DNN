import pandas as pd
# import matplotlib.pyplot as pltp
import numpy as np

raw_train = pd.read_csv('/Users/kylenwilliams/Fashion MNIST/fashion-mnist_train.csv')  # your path to data file
raw_test = pd.read_csv('/Users/kylenwilliams/Fashion MNIST/fashion-mnist_test.csv')  # your path to data file


train_data = np.array(raw_train)  # train data array
test_data = np.array(raw_test)  # test data array

np.random.shuffle(train_data)
np.random.shuffle(test_data)

labels = ['T-Shirt', 'Trousers', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

EPOCHS = 1000
BATCH_SIZE = 8
CLIPPER = 1e-7


class DenseLayer:
    def __init__(self, nrn_inputs, nrn_num):
        self.weights = 0.01 * np.random.randn(nrn_inputs, nrn_num)  # already transposed weights matrix
        self.biases = np.zeros((1, nrn_num))

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
        self.dinputs = np.dot(dvalues, self.weights.T)
        # weights derivative first so it takes the shape of weights
        self.dweights = np.dot(self.inputs.T, dvalues)
        # bias is added del of addition is 1
        # so all we have to do is the adding of the dvalues for each neuron
        # over all the batches
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

    def get_params(self):
        return self.weights, self.biases


class ReLuActivation:
    def forward_pass(self, inputs):
        self.input = inputs
        self.output = np.maximum(0, inputs)

    def backward_pass(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.input <= 0] = 0


class SoftmaxActivation:
    def forward_pass(self, inputs):
        # sub by max to keep the values from exploding
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        normalized_exp_vals = exp_values / np.sum(exp_values)
        self.output = normalized_exp_vals
        return self.output  # without the return I had some NoneType errors so this fixed that

    # we don't actually use this backward pass but its nice to have (we use the combined one)
    # also btw idx means index
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


class CategoricalCrossEntropyLoss(GeneralLoss):
    def forward_pass(self, y_pred, y_true):
        correct_confidences = 0
        samples = len(y_pred)

        # Clip data to prevent log(0)
        # Clip both sides to not drag to anything
        y_pred_clipped = np.clip(y_pred, CLIPPER, 1 - CLIPPER)

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


class AdamOptimizer:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.0001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_optimize(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    def optimize(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * \
                                 layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
                               layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
                                     (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
                                   (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
                             (1 - self.beta_2) * layer.dweights ** 2

        layer.bias_cache = self.beta_2 * layer.bias_cache + \
                           (1 - self.beta_2) * layer.dbiases ** 2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
                                 (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
                               (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) +
                          self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        bias_momentums_corrected / \
                        (np.sqrt(bias_cache_corrected) +
                         self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


# fun colors to make reading the output a little easier
class Color:
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


ground_truths = []
for data_point in train_data:
    ground_truths = np.append(ground_truths, [data_point[0]], 0)     # adds truth values to ground true array
ground_truths = ground_truths.astype(int)
# ground_truths = ground_truths.reshape((7500, BATCH_SIZE))

train_data = np.delete(train_data, 0, axis=1)  # deletes all the truths from input data

# instantiating (what a weird word) the neural net
input_layer = DenseLayer(784, 784)
activation_lay1 = ReLuActivation()

layer_2 = DenseLayer(784, 64)
activation_lay2 = ReLuActivation()
layer_3 = DenseLayer(64, 64)
activation_lay3 = ReLuActivation()
layer_4 = DenseLayer(64, 64)
activation_lay4 = ReLuActivation()

output_layer = DenseLayer(64, 10)
softmax_activation_loss = ActivationSoftmaxLossCategoricalCrossEntropy()

optimizer = AdamOptimizer(decay=5e-7)

for EPOCH in range(EPOCHS):
    # minus BATCH_SIZE because if it chooses like 5999,
    # there aren't ten more after it so it can't calculate anything with that
    clothing_idx = np.random.randint(0, train_data.shape[0] - BATCH_SIZE)

    batch_data = train_data[clothing_idx:clothing_idx + BATCH_SIZE]
    batch_truths = ground_truths[clothing_idx:clothing_idx + BATCH_SIZE]

    # forward pass
    input_layer.forward_pass(batch_data)
    activation_lay1.forward_pass(input_layer.output)

    layer_2.forward_pass(activation_lay1.output)
    activation_lay2.forward_pass(layer_2.output)

    layer_3.forward_pass(activation_lay2.output)
    activation_lay3.forward_pass(layer_3.output)

    layer_4.forward_pass(activation_lay3.output)
    activation_lay4.forward_pass(layer_4.output)

    output_layer.forward_pass(activation_lay4.output)

    softmax_activation_loss.forward_pass(output_layer.output, batch_truths)
    loss = softmax_activation_loss.forward_pass(output_layer.output, batch_truths)

    for BATCH in range(BATCH_SIZE):
        # finds which the neural net predicted
        prediction = np.argmax(softmax_activation_loss.output[BATCH])
        prediction_label = labels[int(prediction)]
        print(f'Batch {BATCH + 1}- guess: {prediction_label} truth: {labels[int(batch_truths[BATCH])]}')
        prob_dist = softmax_activation_loss.output
    print('')

    # accuracy calc
    if len(ground_truths.shape) == 2:
        ground_truths = np.argmax(ground_truths, axis=1)
    accuracy = np.mean(prediction == ground_truths) * 100

    print('\nAverage loss for this batch: ', loss)

    # backward pass is after loss and stuff because it needs to change for next layer
    softmax_activation_loss.backward_pass(softmax_activation_loss.output, batch_truths)
    output_layer.backward_pass(softmax_activation_loss.dinputs)

    activation_lay4.backward_pass(output_layer.dinputs)
    layer_4.backward_pass(activation_lay4.dinputs)

    activation_lay3.backward_pass(layer_4.dinputs)
    layer_3.backward_pass(activation_lay3.dinputs)

    activation_lay2.backward_pass(layer_3.dinputs)
    layer_2.backward_pass(activation_lay2.dinputs)

    activation_lay1.backward_pass(layer_2.dinputs)
    input_layer.backward_pass(activation_lay1.dinputs)

    optimizer.pre_optimize()
    optimizer.optimize(input_layer)
    optimizer.post_update_params()
    optimizer.pre_optimize()
    optimizer.optimize(layer_2)
    optimizer.post_update_params()
    optimizer.pre_optimize()
    optimizer.optimize(layer_3)
    optimizer.post_update_params()
    optimizer.pre_optimize()
    optimizer.optimize(layer_4)
    optimizer.post_update_params()
    optimizer.pre_optimize()
    optimizer.optimize(output_layer)
    optimizer.post_update_params()

    print(f'Accuracy: {accuracy}%')

    # # if you're on the last epoch show that img
    # if EPOCH+1/EPOCHS == 1:
    #     img = train_data[clothing_idx, 1:784].reshape((28, 28))   # reshaping the flattened array to so plt can show it
    #     pltp.imshow(img, cmap='gray')
    #     pltp.show()
