from matplotlib import pyplot as plt
import numpy as np
import pickle

def get_data(inputs_file_path):
    """
    Takes in an inputs file path and labels file path, loads the data into a dict, 
    normalizes the inputs, and returns (NumPy array of inputs, NumPy 
    array of labels). 
    :param inputs_file_path: file path for ONE input batch, something like 
    'cifar-10-batches-py/data_batch_1'
    :return: NumPy array of inputs as float32 and labels as int8
    """
    #TODO: Load inputs and labels
    with open(inputs_file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    X = dict['data']
    Y = dict['labels']
    Y = np.array(Y,"int8")
    #TODO: Normalize inputs
    X = (X/255)
    return X.astype("float32"),Y

class Model:
    """
    This model class will contain the architecture for
    your single layer Neural Network for classifying CIFAR10 with 
    batched learning. Please implement the TODOs for the entire 
    model but do not change the method and constructor arguments. 
    Make sure that your Model class works with multiple batch 
    sizes. Additionally, please exclusively use NumPy and 
    Python built-in functions for your implementation.
    """

    def __init__(self):
        # TODO: Initialize all hyperparametrs
        self.input_size = 3072 # Size of image vectors
        self.num_classes = 10 # Number of classes/possible labels
        self.batch_size = 300
        self.learning_rate = 0.02

        # TODO: Initialize weights and biases
        self.W = np.zeros((self.num_classes,self.input_size),dtype="float32")
        self.b = np.zeros((self.num_classes),dtype="float32")

    def forward(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized (0.0 to 1.0) batch of images,
                       (batch_size x 3072) (2D), where batch can be any number.
        :return: probabilities, probabilities for each class per image # (batch_size x 10)
        """
        # TODO: Write the forward pass logic for your model
        softmax_P = np.empty((0,10),dtype="float32")
        for image_idx in range(0,np.shape(inputs)[0]):
            F_x = self.W.dot(inputs[image_idx]) + self.b
            # Avoid exp overflows
            F_x = F_x - np.amax(F_x)
            softmax_P = np.append(softmax_P,[F_x.T],axis=0)
        # TODO: Calculate, then return, the probability for each class per image using the Softmax equation
        exponents = np.exp(softmax_P)
        sum_of_exponents = np.sum(exponents,axis=1)
        probabilities = exponents/sum_of_exponents[:,None]
        return probabilities
    
    def loss(self, probabilities, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Loss should be generally decreasing with every training loop (step). 
        :param probabilities: matrix that contains the probabilities 
        of each class for each image
        :param labels: the true batch labels
        :return: average loss per batch element (float)
        """
        # TODO: calculate average cross entropy loss for a batch
        crossentropy_loss=0
        for image_idx in range(0,np.shape(probabilities)[0]):
            crossentropy_loss -= np.log(probabilities[image_idx][labels[image_idx]])
        return float(crossentropy_loss/(np.shape(probabilities)[0]))
    
    def compute_gradients(self, inputs, probabilities, labels):
        """
        Returns the gradients for model's weights and biases 
        after one forward pass and loss calculation. You should take the
        average of the gradients across all images in the batch.
        :param inputs: batch inputs (a batch of images)
        :param probabilities: matrix that contains the probabilities of each 
        class for each image
        :param labels: true labels
        :return: gradient for weights,and gradient for biases
        """
        # TODO: calculate the gradients for the weights and the gradients for the bias with respect to average loss
        dW = np.zeros((self.num_classes,self.input_size),dtype="float32")
        dB = np.zeros((self.num_classes),dtype="float32")
        prob_cpy = probabilities.copy()
        for idx in range(0,np.shape(probabilities)[0]):
            prob_cpy[idx,labels[idx]] = prob_cpy[idx,labels[idx]] - 1.0
            dW = dW + inputs[idx]*(prob_cpy[idx,:,None])
            dB = dB + prob_cpy[idx]
        return dW/np.shape(probabilities)[0],dB/np.shape(probabilities)[0]

    
    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number 
        of correct predictions with the correct answers.
        :param probabilities: result of running model.forward() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        # TODO: calculate the batch accuracy
        predictions = np.argmax(probabilities,axis=1)
        count = np.sum(predictions == labels)
        return float(count/np.shape(probabilities)[0])


    def gradient_descent(self, gradW, gradB):
        '''
        Given the gradients for weights and biases, does gradient 
        descent on the Model's parameters.
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        '''
        # TODO: change the weights and biases of the model to descent the gradient
        self.W = self.W - self.learning_rate*gradW
        self.b = self.b - self.learning_rate*gradB
        return
    
def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels.
    :param model: the initialized model to use for the forward 
    pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training)
    :param train_inputs: train labels (all labels to use for training)
    :return: None
    '''
    # TODO: Iterate over the training inputs and labels, in model.batch_size increments
    # TODO: For every batch, compute then descend the gradients for the model's weights
    num_images = np.shape(train_inputs)[0]
    start_idx = 0
    losses = np.empty((0))
    for batch_index in range(0,int(num_images/model.batch_size)):
        end_idx = int((batch_index+1)*model.batch_size)
        if batch_index == int(num_images/model.batch_size)-1:
            end_idx = num_images
        probabilities = model.forward(train_inputs[start_idx:end_idx])
        batchloss = model.loss(probabilities,train_labels[start_idx:end_idx])
        losses = np.append(losses,batchloss)
        W_gradient, B_gradient = model.compute_gradients(train_inputs[start_idx:end_idx],probabilities,train_labels[start_idx:end_idx])
        model.gradient_descent(W_gradient,B_gradient)
        start_idx = end_idx
    # Optional TODO: Call visualize_loss and observe the loss per batch as the model trains.
    return

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. 
    :param test_inputs: CIFAR10 test data (all images to be tested)
    :param test_labels: CIFAR10 test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """
    # TODO: Iterate over the testing inputs and labels
    # TODO: Return accuracy across testing set
    test_probabilities = model.forward(test_inputs)
    return model.accuracy(test_probabilities,test_labels)


def visualize_loss(losses):
    """
    Uses Matplotlib to visualize loss per batch. You can call this in train() to observe.
    param losses: an array of loss value from each batch of train

    NOTE: DO NOT EDIT
    
    :return: doesn't return anything, a plot should pop-up
    """

    plt.ion()
    plt.show()

    x = np.arange(1, len(losses)+1)
    plt.xlabel('i\'th Batch')
    plt.ylabel('Loss Value')
    plt.title('Loss per Batch')
    plt.plot(x, losses, color='r')
    plt.draw()
    plt.pause(0.001)


def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data()
    :param probabilities: the output of model.forward()
    :param image_labels: the labels from get_data()

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    plt.ioff()

    images = np.reshape(image_inputs, (-1, 3, 32, 32))
    images = np.moveaxis(images, 1, -1)
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()

def main():
    '''
    Read in CIFAR10 data, initialize your model, and train and test your model 
    for one epoch. The number of training steps should be your the number of 
    batches you run through in a single epoch. 
    :return: None
    '''
    # TODO: load CIFAR10 train and test examples into train_inputs, train_labels, test_inputs, test_labels
    train_inputs = np.empty((0,3072))
    train_labels = np.empty((0),dtype="int8")
    for batch_number in range(1,6):
        X,Y = get_data('cifar-10-batches-py/data_batch_'+str(batch_number))
        train_inputs = np.append(train_inputs,X,axis=0)
        train_labels = np.append(train_labels,Y,axis=0)
    test_inputs,test_labels = get_data('cifar-10-batches-py/test_batch')
    # TODO: Create Model
    myModel = Model()
    # TODO: Train model by calling train() ONCE on all data
    train(myModel,train_inputs,train_labels)

    # TODO: Test the accuracy by calling test() after running train()
    print(f"{test(myModel,test_inputs,test_labels):.4f}")

    # TODO: Visualize the data by using visualize_results() on a set of 10 examples
    X_visualize = test_inputs[0:10]
    Y_visualize = test_labels[0:10]
    Prob_visualize = myModel.forward(X_visualize)
    visualize_results(X_visualize,Prob_visualize,Y_visualize)
    return
    
if __name__ == '__main__':
    main()
