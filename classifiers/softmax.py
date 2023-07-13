import numpy as np
from random import shuffle

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
    num_train = X.shape[0]
    num_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)  # for numerical stability
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)

        correct_class_prob = probs[y[i]]
        loss -= np.log(correct_class_prob)

        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += (probs[j] - 1) * X[i]
            else:
                dW[:, j] += probs[j] * X[i]

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W
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

    # Compute scores
    scores = np.dot(X, W)

    # Shift scores for numeric stability
    scores -= np.max(scores, axis=1, keepdims=True)

    # Compute softmax probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute loss
    correct_probs = probs[np.arange(num_train), y]
    loss = np.sum(-np.log(correct_probs))
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    # Compute gradient
    dscores = probs
    dscores[np.arange(num_train), y] -= 1
    dW = np.dot(X.T, dscores)
    dW /= num_train
    dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW



# def softmax_loss_vectorized(W, X, y, reg):
#     """
#     Softmax loss function, vectorized version.

#     Inputs and outputs are the same as softmax_loss_naive.
#     """
#     # Initialize the loss and gradient to zero.
#     loss = 0.0
#     dW = np.zeros_like(W)
#     num_train = X.shape[0]

#     #############################################################################
#     # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
#     # Store the loss in loss and the gradient in dW. If you are not careful     #
#     # here, it is easy to run into numeric instability. Don't forget the        #
#     # regularization!                                                           #
#     #############################################################################
#     scores = X @ W
#     # Shift scores for numeric stability
#     scores -= np.max(scores, axis=1, keepdims=True)
#     scores_exp = np.exp(scores)
#     scores_exp_sum = np.sum(scores_exp, axis=1, keepdims=True)
#     probs = scores_exp / scores_exp_sum
#     logits = np.log(probs)
#     neg_log_likelihood = np.array([-logits[j, ix] for j, ix in enumerate(y)])
#     probs_modified = np.copy(probs)
#     probs_modified[(np.arange(num_train), y)] -= 1
#     dW = X.T @ probs_modified
#     loss = np.mean(neg_log_likelihood)
#     loss += 0.5 * reg * np.sum(W * W)
#     dW /= num_train
#     dW += reg * W
#     #############################################################################
#     #                          END OF YOUR CODE                                 #
#     #############################################################################

#     return loss, dW






# # import numpy as np
# # from random import shuffle

# # def softmax_loss_naive(W, X, y, reg):
# #     loss = 0.0
# #     dW = np.zeros_like(W)
# #     # for loop implementation
# #     scores = X @ W
# #     batch_size = X.shape[0]
# #     loss = 0.0
# #     dscores = np.zeros_like(scores)
# #     for i in range(batch_size):
# #         score = scores[i, :]
# #         score_exp = np.exp(score)
# #         score_exp_sum = np.sum(score_exp)

# #         dscore = np.zeros_like(score)
# #         dscore_exp = np.zeros_like(score_exp)
# #         dscore_exp_sum = np.zeros_like(score_exp_sum)

# #         prob = score_exp / score_exp_sum
# #         dprob = np.zeros_like(prob)

# #         logit = np.log(prob)
# #         dlogit = np.zeros_like(logit)

# #         neg_logit = -logit[y[i]]
# #         dneg_logit = 1
# #         dlogit[y[i]] = -dneg_logit 
# #         dprob = (1/prob) * dlogit
# #         dscore_exp = (1/score_exp_sum) * dprob
# #         dscore_exp_sum = (-score_exp / score_exp_sum**2) * dprob
# #         dscore_exp += dscore_exp_sum
# #         dscore = score_exp * dscore_exp
# #         dscores[i, :] = dscore

# #         loss += neg_logit

# #         dW += np.outer(X[i], dscore)

# #     loss /= batch_size
# #     loss += 0.5 * reg * np.sum(W*W)

# #     dW /= batch_size
# #     dW += reg * W

    

# #     return loss, dW

# import numpy as np
# from random import shuffle

# def softmax_loss_naive(W, X, y, reg):
#     loss = 0.0
#     dW = np.zeros_like(W)
#     scores = X @ W
#     batch_size = X.shape[0]
#     dscores = np.zeros_like(scores)
    
#     for i in range(batch_size):
#         score = scores[i, :]
#         score_exp = np.exp(score)
#         score_exp_sum = np.sum(score_exp)

#         dscore = np.zeros_like(score)
#         dscore_exp = np.zeros_like(score_exp)
#         dscore_exp_sum = np.zeros_like(score_exp_sum)

#         prob = score_exp / score_exp_sum
#         dprob = np.zeros_like(prob)

#         logit = np.log(prob)
#         dlogit = np.zeros_like(logit)

#         neg_logit = -logit[y[i]]
#         dneg_logit = 1
#         dlogit[y[i]] = -dneg_logit 
#         dprob = (1/prob) * dlogit
#         dscore_exp = (1/score_exp_sum) * dprob
#         dscore_exp_sum = (-score_exp / score_exp_sum**2) * dprob
#         dscore_exp += dscore_exp_sum
#         dscore = score_exp * dscore_exp
#         dscores[i, :] = dscore

#         loss += neg_logit

#         dW += np.outer(X[i], dscore)

#     loss /= batch_size
#     loss += 0.5 * reg * np.sum(W * W)

#     dW /= batch_size
#     dW += reg * W

#     return loss, dW

import numpy as np
from random import shuffle

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
    - gradient with respect to weights W; an array of the same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)  # for numerical stability
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)

        correct_class_prob = probs[y[i]]
        loss -= np.log(correct_class_prob)

        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += (probs[j] - 1) * X[i]
            else:
                dW[:, j] += probs[j] * X[i]

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W

    return loss, dW

