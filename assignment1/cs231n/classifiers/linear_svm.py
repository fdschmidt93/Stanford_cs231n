import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    count = 0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
            # skip for true class to only loop over incorrect classes
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += X[i] / num_train
                count += 1
        dW[:,y[i]] -= count * X[i] / num_train
        count = 0
            
        
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
    loss /= num_train

  # Add regularization to the loss and gradient.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_train = X.shape[0]
    
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    
    Wx = np.matmul(X, W)
    y_hat = Wx[list(range(num_train)), y]
    
    margin = (Wx.transpose()-y_hat).transpose() + 1
    # set non-contributing rows to zero
    margin[list(range(num_train)), y] = 0
    loss = np.sum(margin[margin > 0]) / num_train + reg * np.sum(W * W) 
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    ## generate binary mask for which non-yi rows have to be added to dW
    count = np.where(margin > 0, 1, 0)
    ## compute negative row sum to scale yi
    count[list(range(num_train)), y]  = -1 * np.sum(count, axis=1)
    ## compute gradient matrix
    dW = np.matmul(X.transpose(), count) / num_train
    ## add regularization
    dW += 2 * reg * W

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW

