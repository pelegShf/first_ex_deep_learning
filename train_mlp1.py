import numpy as np

import mlp1 as mlp
import random
from utils import *
from xor_data import data as xor_dataset

STUDENT={'name': 'Peleg shefi_Daniel bazar',
         'ID': '316523638_314708181'}

def feats_to_vec(features):
    if isinstance(features[-1], (int, float)): # xor
        return np.array(features)
    else:
        if len(features[-1])==1: # unigrams
            F2I_fit = UNI_F2I
            vocab_fit = uni_vocab
        else: # bigrams
            F2I_fit = F2I
            vocab_fit = vocab
        V = np.zeros(len(vocab_fit))
        c = Counter()
        c.update(features)
        d = {k: v for k, v in c.items() if k in vocab_fit}
        for k in d:
            V[F2I_fit[k]] = d[k]
        # Should return a numpy vector of features.
        return V

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        y_hat = mlp.predict(feats_to_vec(features),params)
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
            loss, grads = mlp.loss_and_gradients(x,y,params)
            cum_loss += loss
            # update the parameters according to the gradients
            # and the learning rate.
            params[0] = params[0] - learning_rate * grads[0]  # W
            params[1] = params[1] - learning_rate * grads[1]  # b
            params[2] = params[2] - learning_rate * grads[2]  # U
            params[3] = params[3] - learning_rate * grads[3]  # b_tag

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params

def create_test_result_file(test_dataset,final_params):
    y_hat_preds = []
    for _, features in test_dataset:
        pos = mlp.predict(feats_to_vec(features), final_params)
        # list out keys and values separately
        key_list = list(L2I.keys())
        val_list = list(L2I.values())
        # print key with val 100
        position = val_list.index(pos)
        y_hat_preds.append(key_list[position])

    with open('test.pred','w') as f:
        f.writelines('\n'.join(y_hat_preds))


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    # ...
    in_dim = len(vocab)
    hid_dim = 300
    out_dim = 6
    num_iterations=20
    learning_rate=0.05
    params = mlp.create_classifier(in_dim,hid_dim, out_dim)
    print("letter-bigrams feature set")
    trained_params = train_classifier(TRAIN, DEV, num_iterations, learning_rate, params)
    create_test_result_file(TEST,trained_params)

    print("letter-unigrams feature set")
    in_dim = len(uni_vocab)
    hid_dim = 25
    out_dim = 6
    num_iterations=20
    learning_rate=0.01
    params = mlp.create_classifier(in_dim,hid_dim, out_dim)
    trained_params_unigrams = train_classifier(UNI_TRAIN, UNI_DEV, num_iterations, learning_rate, params)

    print("learning the XOR function (no validation)")
    in_dim = 2
    hid_dim = 4
    out_dim = 2
    num_iterations=30
    learning_rate=0.1
    params = mlp.create_classifier(in_dim,hid_dim, out_dim)
    trained_params_xor = train_classifier(xor_dataset, xor_dataset, num_iterations, learning_rate, params)
