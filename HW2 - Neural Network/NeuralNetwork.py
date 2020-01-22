import numpy as np
import pandas as pd
import random
import csv
import sys
import time

start_time = time.time()

#Extracting Train data and Test data path from Command Line Arguments
no_of_cli_arguments = len(sys.argv)
if no_of_cli_arguments!=3:
    print("Train data and Test data paths are not specified; Please check the CLI arguments")
    sys.exit(0)
args = sys.argv
train_df_path = args[1]
test_df_path = args[2]
indice = train_df_path.index("blackbox")
output_name = "blackbox2{}_predictions.csv".format(train_df_path[indice+9])  #OutputFile Name
df = pd.read_csv(r"{}".format(train_df_path), sep = ',', names=['A','B','C','Output']) #Train data
#One-hot encoding
labels_train_df = pd.get_dummies(df.Output)
#Train Data without Labels
inputs_train_df = df.drop(["Output"],axis=1)


#Relu Activation Function
def relu(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X

#Derivative of Relu
def relu_derivative(X):
    X[X<=0] = 0
    X[X>0] = 1
    return X

# Softmax Activation Function
def softmax(X):
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]
    return X

#Calculate Accuracy
# def accuracy(labels,predictions):
#     accuracy_temp = predictions == labels
#     accuracy = accuracy_temp.mean()
#     return accuracy

#Log loss - for calculating loss/error
def log_loss(y_true, y_prob):
    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    return - np.multiply(y_true, np.log(y_prob)).sum() / y_prob.shape[0]


# Adam Optimizer
class AdamOptimizer():

    def __init__(self, params, learning_rate_init=0.001, beta_1=0.9,
                 beta_2=0.999, epsilon=1e-8):

        self.params = [param for param in params]
        self.learning_rate_init = float(learning_rate_init)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.ms = [np.zeros_like(param) for param in params]
        self.vs = [np.zeros_like(param) for param in params]

    def update_params(self, grads):
        updates = self._get_updates(grads)
        for param, update in zip(self.params, updates):
            param += update

    def _get_updates(self, grads):
        self.t += 1
        self.ms = [self.beta_1 * m + (1 - self.beta_1) * grad
                   for m, grad in zip(self.ms, grads)]
        self.vs = [self.beta_2 * v + (1 - self.beta_2) * (grad ** 2)
                   for v, grad in zip(self.vs, grads)]
        self.learning_rate = (self.learning_rate_init *
                              np.sqrt(1 - self.beta_2 ** self.t) /
                              (1 - self.beta_1 ** self.t))
        updates = [-self.learning_rate * m / (np.sqrt(v) + self.epsilon)
                   for m, v in zip(self.ms, self.vs)]
        return updates



#Input parameters
learning_rate = 0.1
N = labels_train_df.size
no_of_samples = inputs_train_df.shape[0]
input_layer_nodes_count = 3
hidden_layer_1_nodes_count = 20
hidden_layer_2_nodes_count = 20
output_layer_nodes_count = labels_train_df.shape[1]
epochs = 1400
batch_size = 5000
batches = no_of_samples // batch_size
start=0
np.random.seed(10)


#Xavier initialization for weights and bias
weights_1 = np.random.randn(input_layer_nodes_count, hidden_layer_1_nodes_count) * np.sqrt(2/input_layer_nodes_count)
weights_2 = np.random.randn(hidden_layer_1_nodes_count, hidden_layer_2_nodes_count) * np.sqrt(2/hidden_layer_1_nodes_count)
weights_3 = np.random.randn(hidden_layer_2_nodes_count, output_layer_nodes_count) * np.sqrt(2/hidden_layer_2_nodes_count)

bias_1 = np.random.randn(hidden_layer_1_nodes_count) * np.sqrt(2/hidden_layer_1_nodes_count)
bias_2 = np.random.randn(hidden_layer_2_nodes_count) * np.sqrt(2/hidden_layer_2_nodes_count)
bias_3 = np.random.randn(output_layer_nodes_count) * np.sqrt(2/output_layer_nodes_count)

#Initializing AdamOptimizer
adam = AdamOptimizer([weights_1,weights_2,weights_3,bias_1,bias_2,bias_3])
# result = {"Accuracy":[]}

# Core logic for Neural Network
# Dividing the total samples into batches and running FeedForward and BackPropagation logic for each batch
for batch in range(batches):

    # To include last sample as [:n] gives upto n-1
    if batch == batches-1:
        batch_size+=1

    inputs_train = inputs_train_df[start : start+batch_size].values    #Input data for the current batch without labels
    labels_train = labels_train_df[start : start+batch_size].values    #Labels for the current batch
    no_of_samples = inputs_train.shape[0]                              #Total number of samples

    # Looping for defined epochs/rounds
    for epoch in range(epochs):
        #Feed Forward

        #Input to hidden layer 1
        hidden_layer_1_inputs = np.dot(inputs_train,weights_1) + bias_1            # Inputs.weights_1 + bias1
        hidden_layer_1_outputs = relu(hidden_layer_1_inputs)                       # Relu Activation for hidden layer 1

        #hidden layer 1 to hidden layer 2
        hidden_layer_2_inputs = np.dot(hidden_layer_1_outputs,weights_2) + bias_2  # HiddenLayer1.weights_2 + bias2
        hidden_layer_2_outputs = relu(hidden_layer_2_inputs)                       # Relu Activation for hidden layer 2

        #hidden layer 1 to output layer
        output_layer_inputs = np.dot(hidden_layer_2_outputs,weights_3) + bias_3    # HiddenLayer2.weights_3 + bias3
        output_layer_outputs = softmax(output_layer_inputs)                        # Softmax Activation for hidden layer 3

        # Replacing 0 with very small number for output node to avoid log(0) error as it is undefined
        output_layer_outputs[output_layer_outputs == 0.0] = 10**-10

        #Accuracy
        # acc = accuracy(output_layer_outputs,labels_train)
        # result["Accuracy"].append(acc)

        #Back Propagation

        #Finding derivative of error w.r.t weights_3
        output_layer_error = output_layer_outputs - labels_train
        output_layer_delta = output_layer_error
        weights_3_update =  np.dot(hidden_layer_2_outputs.T, output_layer_delta)

        #Finding derivative of error w.r.t bias_3
        bias_3_update = output_layer_delta.mean(axis=0)


        #Finding derivative of error w.r.t weights_2
        hidden_layer_2_error = np.dot(output_layer_delta,weights_3.T)
        hidden_layer_2_delta = hidden_layer_2_error * relu_derivative(hidden_layer_2_inputs)
        weights_2_update = np.dot(hidden_layer_1_outputs.T,hidden_layer_2_delta)

        #Finding derivative of error w.r.t bias_2
        bias_2_update = hidden_layer_2_delta.mean(axis=0)

        #Finding derivative of error w.r.t weights_1
        hidden_layer_1_error = np.dot(hidden_layer_2_delta,weights_2.T)
        hidden_layer_1_delta = hidden_layer_1_error * relu_derivative(hidden_layer_1_inputs)
        weights_1_update = np.dot(inputs_train.T,hidden_layer_1_delta)

        #Finding derivative of error w.r.t bias_1
        bias_1_update = hidden_layer_1_delta.mean(axis=0)

        #Updating the weights and bias using Adams Optimizer
        adam.update_params([weights_1_update,weights_2_update,weights_3_update, bias_1_update,bias_2_update,bias_3_update])

    #Updating the batch
    start = start + batch_size


# Reading Test File
test_df = pd.read_csv(r"{}".format(test_df_path), sep = ',', names=['A','B','C'])
#Converting Pandas DataFrame to Numpy Array
inputs_test = test_df.values


#Feed Forward for Test data

#Input layer to Hidden layer 1
hidden_layer_1_inputs = np.dot(inputs_test,weights_1) + bias_1
hidden_layer_1_outputs = relu(hidden_layer_1_inputs)

#Hidden layer 1 to Hidden layer 2
hidden_layer_2_inputs = np.dot(hidden_layer_1_outputs,weights_2) + bias_2
hidden_layer_2_outputs = relu(hidden_layer_2_inputs)

#Hidden layer 2 to Output layer
output_layer_inputs = np.dot(hidden_layer_2_outputs,weights_3) + bias_3
output_layer_outputs = softmax(output_layer_inputs)
predicted_test_labels = output_layer_outputs.argmax(axis=1)

#Writing Ouput i.e. predicted lables to a csv file
with open(output_name, 'w',newline='') as f:
    writer = csv.writer(f)
    for val in predicted_test_labels:
        writer.writerow([val])

#print("--- %s seconds ---" % (time.time() - start_time))
