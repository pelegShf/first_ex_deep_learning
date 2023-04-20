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
            # print(grads[1].shape)

            params[0] = params[0] - learning_rate*grads[0] #W
            params[1] = params[1] - learning_rate*grads[1] #b

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params

def create_test_result_file(test_dataset,final_params):
    y_hat_preds = []
    for _, features in test_dataset:
        pos = ll.predict(feats_to_vec(features), final_params)
        # list out keys and values separately
        key_list = list(L2I.keys())
        val_list = list(L2I.values())
        # print key with val 100
        position = val_list.index(pos)
        y_hat_preds.append(key_list[position])
    print(y_hat_preds)

    with open('test.pred','w') as f:
        f.writelines('\n'.join(y_hat_preds))


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    
    # ...
    in_dim = 1000
    out_dim = 10
    num_iterations= 10
    learning_rate=0.1
    params = ll.create_classifier(in_dim, out_dim)
    print("letter-bigrams feature set")
    trained_params = train_classifier(TRAIN, DEV, num_iterations, learning_rate, params)
    print("letter-unigrams feature set")
    trained_params_unigrams = train_classifier(UNI_TRAIN, UNI_DEV, num_iterations, learning_rate, params)
    # create_test_result_file(TEST,trained_params)
