import pandas as pd
import matplotlib.pyplot as pltp
import numpy as np

raw_train = pd.read_csv('/Users/kylenwilliams/Fashion MNIST/fashion-mnist_train.csv')  # Your path to data file
raw_test = pd.read_csv('/Users/kylenwilliams/Fashion MNIST/fashion-mnist_test.csv')  # Your path to data file

train_data = np.array(raw_train)  # train data array
test_data = np.array(raw_test)  # test data array

labels = ['T-Shirt', 'Trousers', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

E = np.e
EPOCHS = 5
BATCH_SIZE = 2
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
        exp_values = E ** (inputs - np.max(inputs, axis=1, keepdims=True))  # sub by max to keep the vals from exploding
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
                y_true
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
    def forward_pass(self, grad):
        pass


# fun colors to make reading the output a little easier
class Color:
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


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


choice_frequency_dict = {'T-Shirt': 0, 'Trousers': 0, 'Pullover': 0, 'Dress': 0, 'Coat': 0,
                         'Sandal': 0, 'Shirt': 0, 'Sneaker': 0, 'Bag': 0, 'Ankle Boot': 0}

for EPOCH in range(EPOCHS):
    clothing_idx = np.random.randint(0, train_data.shape[0] - BATCH_SIZE)  # -BATCH_SIZE because if it chooses like
    # 5999, there aren't ten more after it so it can't calculate anything with that

    # # puts all the truths into an array
    ground_truths = []
    # I did this so I can safely modify the inputs
    train_data_4_batch = train_data[clothing_idx:clothing_idx + BATCH_SIZE]

    count = 0
    # I was going to convert train data to a list then .pop()
    # but .pop() only takes one value but we need to index 2, the sample and the truth value (which is at idx 0)
    for data_point in train_data_4_batch:
        ground_truths = np.append(ground_truths, [data_point[0]], 0)     # adds truth values to ground true array

    train_data_4_batch = np.delete(train_data_4_batch, 0, axis=1)  # deletes all the truths from input data
    ground_truths = ground_truths.astype(int)

    print(Color.BLUE + f'Epoch {EPOCH + 1}:' + Color.END)
    for BATCH in range(BATCH_SIZE):
        # truth value for our specific data point
        true_idx = ground_truths[BATCH]
        true_label = labels[true_idx]

        # forward pass
        input_layer.forward_pass(train_data_4_batch)
        activation_lay1.forward_pass(input_layer.output)

        layer_2.forward_pass(activation_lay1.output)
        activation_lay2.forward_pass(layer_2.output)

        layer_3.forward_pass(activation_lay2.output)
        activation_lay3.forward_pass(layer_3.output)

        layer_4.forward_pass(activation_lay3.output)
        activation_lay4.forward_pass(layer_4.output)

        output_layer.forward_pass(activation_lay4.output)

        loss = softmax_activation_loss.forward_pass(output_layer.output, ground_truths)

        print('Index of clothing item:', clothing_idx)

        # finds which the neural net predicted
        prediction = np.argmax(softmax_activation_loss.output)
        prediction_label = labels[int(prediction)]
        prob_dist = softmax_activation_loss.output[BATCH]

        print(Color.BOLD + f'{BATCH + 1}- ' + Color.END + f'Guess: {prediction_label}, Truth: {true_label}')
        # print(f'Probability distribution across labels: \n {prob_dist}\n')

        choice_frequency_dict.update({prediction_label: choice_frequency_dict[prediction_label] + 1})

        # accuracy calc
        # converts data to sparse if one hot again, our data is sparse but I gotta have that delicious scalability
        if len(ground_truths.shape) == 2:
            ground_truths = np.argmax(ground_truths, axis=1)
        accuracy = np.mean(prediction == ground_truths) * 100

        clothing_idx += 1

    choice_probability_dict = choice_frequency_dict.copy()

    # updates dict of labels with percentages of how often they are chosen
    total_iters = sum(choice_probability_dict.values())
    for label, val in choice_frequency_dict.items():
        choice_probability_dict.update({label: f'{(val / total_iters) * 100}%'})

    print('\nAverage loss for this batch: ', loss)
    print(f'Accuracy: {accuracy}%')

    print('Choice frequency: ', choice_frequency_dict)
    # in the end this should be about even because the data has an equal number of each clothing type
    print(f'Choice probabilistic frequency: {choice_probability_dict}\n')

    # backward pass is after loss and stuff because it needs to change for next layer
    softmax_activation_loss.backward_pass(softmax_activation_loss.output, ground_truths)
    output_layer.backward_pass(softmax_activation_loss.dinputs)

    activation_lay4.backward_pass(output_layer.dinputs)
    layer_4.backward_pass(activation_lay4.dinputs)

    activation_lay3.backward_pass(layer_4.dinputs)
    layer_3.backward_pass(activation_lay3.dinputs)

    activation_lay2.backward_pass(layer_3.dinputs)
    layer_2.backward_pass(activation_lay2.dinputs)

    activation_lay1.backward_pass(layer_2.dinputs)
    input_layer.backward_pass(activation_lay1.dinputs)

    # if you're on the last epoch show that img
    if EPOCH+1/EPOCHS == 1:
        # its 1:785 b/c the 0th val is the ground truth
        img = train_data[clothing_idx, 1:785].reshape((28, 28))   # reshaping the flattened array to so plt can show it
        pltp.imshow(img, cmap='gray')
        pltp.show()
