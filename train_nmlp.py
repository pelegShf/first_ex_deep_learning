import numpy as np

import mlpn as mlpn
import random
from utils import *
from xor_data import data as xor_dataset

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def feats_to_vec(features):
    if isinstance(features[-1], (int, float)):
        return np.array(features)
    else:
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
        y_hat = mlpn.predict(feats_to_vec(features),params)
        comparing_label = label if isinstance(label, (int, float)) else L2I[label]
        if y_hat == comparing_label:
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
            y = label if isinstance(label, (int, float)) else L2I[label] # convert the label to number if needed.
            loss, grads = mlpn.loss_and_gradients(x,y,params)
            # print(params)
            cum_loss += loss
            # update the parameters according to the gradients
            # and the learning rate.
            layers_count = int(len(params) / 2)

            for l in range(1, layers_count):
                # print(2 * (l - 1))
                # print(params[2 * (l - 1)].shape)
                # print(grads[f"dW{l}"].shape)

                params[2 * (l - 1)] -= learning_rate * grads[f"dW{l}"]
                params[2 * (l - 1) + 1] -= learning_rate * grads[f"db{l}"]
            # params[0] = params[0] - learning_rate * grads[0]  # W
            # params[1] = params[1] - learning_rate * grads[1]  # b
            # params[2] = params[2] - learning_rate * grads[2]  # U
            # params[3] = params[3] - learning_rate * grads[3]  # b_tag

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params

def create_test_result_file(test_dataset,final_params):
    y_hat_preds = []
    for _, features in test_dataset:
        pos = mlpn.predict(feats_to_vec(features), final_params)
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
    hid_dim = 500
    hid_dim2 = 500
    hid_dim3 = 20
    out_dim = 6
    num_iterations= 40
    learning_rate=0.1
    # params = mlpn.create_classifier([in_dim,hid_dim,hid_dim2, out_dim])
    print("letter-bigrams feature set")
    # trained_params = train_classifier(TRAIN, DEV, num_iterations, learning_rate, params)

    # XOR
    in_dim = 2
    hid_dim = 4
    out_dim = 2
    num_iterations = 50
    learning_rate = 0.5
    params = mlpn.create_classifier([in_dim, hid_dim, out_dim])
    trained_params_xor = train_classifier(xor_dataset, xor_dataset, num_iterations, learning_rate, params)
