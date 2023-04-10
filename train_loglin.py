import numpy as np

import loglinear as ll
import random
from utils import *

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def feats_to_vec(features):
    V = np.zeros(len(vocab))
    c = Counter()
    c.update(features)
    d = {k: v for k, v in c.items() if k in vocab}
    for k in d:
        V[F2I[k]] = d[k]
    # Should return a numpy vector of features.
    return V

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        y_hat = ll.predict(feats_to_vec(features),params)
        if y_hat == L2I[label]:
            good = good + 1
        else:
            bad = bad + 1
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)

    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = L2I[label]                 # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            # update the parameters according to the gradients
            # and the learning rate.
            params[0] = params[0] - learning_rate*grads[0] #W
            params[1] = params[1] - learning_rate*grads[1] #b


        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    
    # ...
    in_dim = 600
    out_dim = 10
    num_iterations= 10
    learning_rate=0.1
    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(TRAIN, DEV, num_iterations, learning_rate, params)
