import numpy as np

STUDENT={'name': 'Peleg shefi_Daniel bazar',
         'ID': '316523638_314708181'}

def classifier_output(x, params):
    layers_count = int(len(params) / 2)
    x_copy = np.array(x)  # (in_dim,)
    x_copy = x_copy.reshape(x_copy.shape[0], 1)  # (in_dim,1)

    # forward

    cache = {"H0": x_copy}
    for layer in range(1, layers_count + 1):
        w = params[2 * (layer - 1)]
        b = params[2 * (layer - 1) + 1]
        cache[f"Z{layer}"] = np.dot(w.T, cache[f"H{layer - 1}"]) + b
        if layer == layers_count:
            cache[f"H{layer}"] = softmax(cache[f"Z{layer}"])
        else:
            cache[f"H{layer}"] = np.tanh(cache[f"Z{layer}"])
    probs = cache[f"Z{layers_count}"]
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    layers_count = int(len(params) / 2)
    x_copy = np.array(x)  # (in_dim,)
    x_copy = x_copy.reshape(x_copy.shape[0], 1)  # (in_dim,1)

    #forward
    cache = {"H0": x_copy}
    for layer in range(1,layers_count+1):
        w = params[2*(layer-1)]
        b = params[2*(layer-1)+1]
        cache[f"Z{layer}"] = np.dot(w.T,cache[f"H{layer-1}"]) + b
        if layer == layers_count:
            cache[f"Z{layer}"] = softmax(cache[f"Z{layer}"])
        else:
            cache[f"H{layer}"] = np.tanh(cache[f"Z{layer}"])
    #loss
    y_hat = cache[f"Z{layers_count}"]
    loss = -1 * np.log(y_hat[y])[0]  # scalar
    y_true = np.zeros((y_hat.shape[0], 1))  # (out_dim,1)
    y_true[y] = 1

    #back
    grads = {}
    dH = y_hat-y_true
    for l in reversed(range(1, layers_count+1)):
        W = params[2 * (l - 1)]
        b = params[2 * (l - 1) + 1]
        if l == layers_count:
            grads[f"dZ{l}"] = y_hat - y_true
        else:
            grads[f"dZ{l}"] = dH * (1 - cache[f"H{l}"] ** 2)
        dH = np.dot(W, grads[f"dZ{l}"])
        grads[f"dW{l}"] = np.dot(cache[f"H{l - 1}"],grads[f"dZ{l}"].T)
        grads[f"db{l}"] = grads[f"dZ{l}"]

    return loss, grads

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    np.random.seed(42)
    params = []
    for i in range(len(dims)-1):
        w_i = np.random.randn(dims[i], dims[i+1])
        b_i = np.random.randn(dims[i+1], 1)
        params.append(w_i)
        params.append(b_i)
    return params

def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()



if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    [W, b,W2,b2, W3,b3] = create_classifier([3, 5,6,4])


    def _loss_and_W_grad(W):
        global b
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, W2,b2,W3,b3])
        return loss, grads["dW1"]

    def _loss_and_b_grad(b):
        global W
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, W2,b2,W3,b3])
        return loss, grads["db1"]

    def _loss_and_W2_grad(W2):
        global b2
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, W2,b2,W3,b3])
        return loss, grads["dW2"]

    def _loss_and_b2_grad(b2):
        global W2
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, W2,b2,W3,b3])
        return loss, grads["db2"]

    def _loss_and_W3_grad(W3):
        global b3
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, W2,b2,W3,b3])
        return loss, grads["dW3"]

    def _loss_and_b3_grad(b3):
        global W3
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, W2,b2,W3,b3])
        return loss, grads["db3"]


    for _ in range(10):
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0],1)
        W2 = np.random.randn(W2.shape[0], W2.shape[1])
        b2 = np.random.randn(b2.shape[0], 1)
        W3 = np.random.randn(W3.shape[0], W3.shape[1])
        b3 = np.random.randn(b3.shape[0],1)
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_W2_grad, W2)
        gradient_check(_loss_and_b2_grad, b2)
        gradient_check(_loss_and_W3_grad, W3)
        gradient_check(_loss_and_b3_grad, b3)



