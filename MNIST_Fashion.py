import pandas as pd
import matplotlib.pyplot as pltp
import numpy as np

raw_train = pd.read_csv('/Users/kylenwilliams/Fashion MNIST/fashion-mnist_train.csv')   # Your path to data file
raw_test = pd.read_csv('/Users/kylenwilliams/Fashion MNIST/fashion-mnist_test.csv')     # Your path to data file

train_data = np.array(raw_train)    # train data array
test_data = np.array(raw_test)      # test data array

labels = ['T-Shirt', 'Trousers', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

E = np.e
EPOCHS = 1
BATCH_SIZE = 8
BETA = 1e-7


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
        self.dweights = np.dot(self.inputs.T, dvalues)   # weights derivative first so it takes the shape of weights
        # bias is added del of addition is 1, so all we have to do is the adding of the dvalues for each neuron
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
        self.dinputs[self.inputs <= 0] = 0


class SoftmaxActivation:
    def forward_pass(self, inputs):
        exp_values = E ** (inputs - np.max(inputs, axis=1, keepdims=True))  # sub by max to keep the vals from exploding
        normalized_exp_vals = exp_values / np.sum(exp_values)
        self.output = normalized_exp_vals

    def get_params(self):
        return self.outputs


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
        y_pred_clipped = np.clip(y_pred, BETA, 1 - BETA)

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


# combined for faster back prop
class ActivationSoftmaxLossCategoricalCrossEntropy:
    def init(self):
        self.activation = SoftmaxActivation()
        self.loss = CategoricalCrossEntropyLoss()

    def forward_pass(self, inputs, y_true):
        self.output = self.activation.forward_pass(inputs).output

        return self.loss.calculate(self.output, y_true)

    def backward_pass(self, dvalues, y_true):
        samples = len(dvalues)

        self.dinputs = dvalues.copy()

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs[range(samples), y_true] -= 1  # the derivative of both combined is simplified down to subtracting 1
        self.dinputs = self.dinputs/samples


class AdamOptimizer:
    def forward_pass(self, grad):
        pass

    # no backward pass for adam because its fixing the net not actually part of it


# instantiating (what a weird word) the neural net

input_layer = DenseLayer(785, 785)
actviation_lay1 = ReLuActivation()

layer_2 = DenseLayer(785, 64)
actviation_lay2 = ReLuActivation()
layer_3 = DenseLayer(64, 64)
actviation_lay3 = ReLuActivation()
layer_4 = DenseLayer(64, 64)
actviation_lay4 = ReLuActivation()

loss_obj = CategoricalCrossEntropyLoss()

output_layer = DenseLayer(64, 10)
output_activation = SoftmaxActivation()

choice_frequency_dict = {'T-Shirt': 0, 'Trousers': 0, 'Pullover': 0, 'Dress': 0, 'Coat': 0,
                         'Sandal': 0, 'Shirt': 0, 'Sneaker': 0, 'Bag': 0, 'Ankle Boot': 0}


for EPOCH in range(EPOCHS):
    clothing_idx = np.random.randint(0, train_data.shape[0]-BATCH_SIZE)  # -BATCH_SIZE because if it chooses like
    # 5999, there aren't ten more after it so it can't calculate anything with that

    # puts all the truths into an array
    ground_truths = []
    for data_point in range(BATCH_SIZE):
        current_sample = train_data[clothing_idx + data_point]
        ground_truths.append(current_sample[0])
    ground_truths = np.array(ground_truths)

    print(f'Epoch {EPOCH + 1}:')
    for BATCH in range(BATCH_SIZE):
        # forward pass
        input_layer.forward_pass(train_data[clothing_idx:clothing_idx + BATCH_SIZE])
        actviation_lay1.forward_pass(input_layer.output)

        layer_2.forward_pass(actviation_lay1.output)
        actviation_lay2.forward_pass(layer_2.output)

        layer_3.forward_pass(actviation_lay2.output)
        actviation_lay3.forward_pass(layer_3.output)

        layer_4.forward_pass(actviation_lay3.output)
        actviation_lay4.forward_pass(layer_4.output)

        output_layer.forward_pass(actviation_lay4.output)    # The original output will look skewed slightly to
        output_activation.forward_pass(output_layer.output)  # shirt/sandal, but that's simply because they are
        # squarely in the middle of the array, and the probability distribution is normal

        print(f'Index of clothing item: {clothing_idx}')

        # finds which the neural net predicted
        prediction = np.max(output_activation.output)
        prediction_label = labels[int(np.argmax(output_activation.output))]

        # ground_truth = train_data[clothing_idx, 0]
        true_idx = train_data[clothing_idx, 0]
        true_label = labels[true_idx]
        prob_dist = output_activation.output[BATCH]

        loss = loss_obj.forward_pass(output_activation.output, ground_truths)

        print(f'{BATCH+1}- Guess: {prediction_label}, Truth: {true_label}')
        print(f'Probability distribution across labels: \n {prob_dist}')
        print(f'Loss: {loss[BATCH]}')
        print('')
        clothing_idx += 1

        choice_frequency_dict.update({prediction_label: choice_frequency_dict[prediction_label] + 1})

    choice_probability_dict = choice_frequency_dict.copy()

    # updates dict of labels with percentage of how often they are chosen
    total_iters = sum(choice_probability_dict.values())
    for label, val in choice_frequency_dict.items():
        choice_probability_dict.update({label: f'{(val/total_iters)*100}%'})

    print(f'Average loss for this batch: {loss_obj.calculate(output_activation.output, np.array(ground_truths))}')
    print(f'Choice frequency: {choice_frequency_dict}')

    # in the end this should be about even because the data has an equal number of each clothing type
    print(f'Choice probabilistic frequency: {choice_probability_dict}')

    # backward pass

    # if EPOCH+1/EPOCHS == 1:
    #     img = train_data[clothing_idx, 0:784].reshape((28, 28))   # reshaping the flattened array to so plt can show it
    #     pltp.imshow(img, cmap='gray')
    #     pltp.show()
