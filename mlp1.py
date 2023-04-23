import numpy as np

STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()


def classifier_output(x, params):
    W, b, U, b_tag = params
    x_copy = x.copy()
    x_copy = x_copy.reshape(x_copy.shape[0],1)
    z1 = np.dot(W.T,x_copy) + b
    h1 = np.tanh(z1)
    Z2 = np.dot(U.T,h1) + b_tag
    probs = softmax(Z2)
    return probs


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    # W - (in_dim,hid_dim)
    # b - (hid_dim,1)
    # U - (hid_dim,out_dim)
    # b_tag - (out_dim,1)
    #x_copy - (in_dim,1)
    W, b, U, b_tag = params
    x_copy = np.array(x) #(in_dim,)
    x_copy = x_copy.reshape(x_copy.shape[0],1) #(in_dim,1)

    z1 = np.dot(W.T,x_copy) + b #(hid_dim,1)
    h1 = np.tanh(z1)#(hid_dim,1)
    z2 = np.dot(U.T, h1) + b_tag #(out_dim,1)

    y_hat = softmax(z2) #(out_dim,1)
    loss = -1 * np.log(y_hat[y]) #scalar
    # print(-1 * np.log(y_hat[y]))
    y_true = np.zeros((y_hat.shape[0],1))  #(out_dim,1)
    y_true[y] = 1

    gZ2 = (y_hat - y_true) #(out_dim,1)
    gU = np.dot(h1.reshape(h1.shape[0],1), gZ2.T) #(hid_dim,out_dim)
    gb_tag = gZ2 #(out_dim,1)
    gh1 = np.dot(U, gZ2) #(hid_dim,1)
    deriv = (1 - np.power(np.tanh(z1), 2))
    gZ1 = gh1 * deriv #(hid_dim,1)
    gW = np.dot(x_copy.reshape(x_copy.shape[0],1), gZ1.T) #(in_dim,hid_dim)
    gb = gZ1 #(hid_dim,1)



    return loss, [gW, gb, gU, gb_tag]

def tanh_derivative(z1):
    return (1 - np.power(np.tanh(z1), 2))
def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    params = []
    np.random.seed(42)

    W = np.random.randn(in_dim, hid_dim)*0.05
    b = np.random.randn(hid_dim,1)*0.05
    U = np.random.randn(hid_dim, out_dim)*0.05
    b_tag = np.random.randn(out_dim,1)*0.05

    params = [W, np.asarray(b), U, np.asarray(b_tag)]
    return params


if __name__ == '__main__':
    # Sanity checks for softmax. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.
    test1 = softmax(np.array([1, 2]))
    print(test1)
    assert np.amax(np.fabs(test1 - np.array([0.26894142, 0.73105858]))) <= 1e-6

    test2 = softmax(np.array([1001, 1002]))
    print(test2)
    assert np.amax(np.fabs(test2 - np.array([0.26894142, 0.73105858]))) <= 1e-6

    test3 = softmax(np.array([-1001, -1002]))
    print(test3)
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6

    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    [W, b, U,b_tag] = create_classifier(3, 5,4)


    def _loss_and_W_grad(W):
        global b
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U,b_tag])
        return loss, grads[0]


    def _loss_and_b_grad(b):
        global W
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U,b_tag])
        return loss, grads[1]

    def _loss_and_u_grad(U):
        global b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U,b_tag])
        return loss, grads[2]
    def _loss_and_b_tag_grad(b_tag):
        global U
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U,b_tag])
        return loss, grads[3]


    for _ in range(10):
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0],1)
        U = np.random.randn(U.shape[0], U.shape[1])
        b_tag = np.random.randn(b_tag.shape[0],1)
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_u_grad, U)
        gradient_check(_loss_and_b_tag_grad, b_tag)



