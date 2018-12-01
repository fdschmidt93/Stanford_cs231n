import numpy as np
from random import shuffle


def softmax(W, X):
    Wx = np.matmul(X, W)
    ## stabilize exponent by deducting row-wise maximum
    Wx -= np.max(Wx, axis = -1, keepdims = True)
    ## axis = -1: always take row sum (irrespective vec or mat)
    ## keepdims: fix broadcasting
    return np.exp(Wx) / np.sum(np.exp(Wx), axis = -1, keepdims = True)

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    for i in range(num_train):
        p = softmax(W, X[i])
        loss -= np.log(p[y[i]])
        p[y[i]] -= 1
        dW += np.matmul(np.matrix(X[i]).T, np.matrix(p)) 
        
        
    # Average loss & gradients
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss and gradient.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    p = softmax(W, X)
    
    # Sum logarithmic deviations for true labels
    loss -= np.sum(np.log(p[range(num_train), y]))
    # Generate gradients
    dscores = p
    dscores[range(num_train), y] -= 1
    dW = np.matmul(X.T, dscores)
    
    
    # Average loss and gradients
    loss /= num_train
    dW /= num_train
    
    # Add regularization loss and gradients
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW