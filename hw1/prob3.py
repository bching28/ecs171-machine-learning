import numpy as np
from math import exp
import time
#import torch

# load training data
#x = torch.from_numpy(np.loadtxt('mnist_X_train.dat'))
#w = torch.DoubleTensor(780,1).uniform_(-0.01, 0.01)
#w = -0.01-0.01 * torch.rand(780,1) + 0.01

# declaring global variables
global train_w
global train_x
global train_y
global test_x
global test_y

# initializing weights with uniform distribution
# from -0.01 to +0.01
train_w = np.random.uniform(-0.01, 0.01, 780)
# importing training data
train_x = np.loadtxt('mnist_X_train.dat')
train_y = np.loadtxt('mnist_y_train.dat')

#importing testing data
test_x = np.loadtxt('mnist_X_test.dat')
test_y = np.loadtxt('mnist_y_test.dat')

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def testDataAccuracy(test_w, test_x, test_y, alpha):
    prediction = 0
    for n in range(0, np.size(test_y)):
        if (np.sign(np.matmul(test_x[n], np.transpose(test_w))) == test_y[n]):
            prediction += 1
        else:
            prediction += -1

    test_accuracy = prediction / float(np.size(test_y)) * 100

    print 'Accuracy of test data:', test_accuracy, "with a step size of", alpha

def gradientDescent():
    # declaring global variables
    global train_w
    global train_x
    global train_y
    global test_x
    global test_y

    # initializing variables
    N = 10000
    alpha = 0.0001
    epoch = 0 # number of iterations
    prediction = 0

    # numerator of the gradient function
    resize_y = np.transpose([train_y, ]*780)
    gradient_yx = resize_y*train_x

    while (True):
        epoch += 1
        prediction = 0
        yxw = train_y * np.matmul(train_x, np.transpose(train_w))
        min_function = np.sum( np.log( 1 + np.exp(-yxw) ) ) / N

        # computer gradient
        resize_yxw = np.transpose([yxw, ]*780)
        gradient = np.sum( np.divide(-gradient_yx, (1+np.exp(resize_yxw))), axis=0 ) / N
        print 'Shape:', np.shape(gradient)
        train_w -= alpha * gradient

        for n in range(0, N):
            if (np.sign(np.matmul(train_x[n], np.transpose(train_w))) == train_y[n]):
                prediction += 1
            else:
                prediction += -1

        train_accuracy = prediction / float(N) * 100

        print 'Epoch:', epoch, '|| Min Function:', min_function, '|| Accuracy:', train_accuracy, '|| Gradient:', np.sum(gradient)

        if (train_accuracy > 99.5): # takes about 2000 epochs
            break

    test_w = train_w # passing in w for test...easier to follow
    testDataAccuracy(test_w, test_x, test_y, alpha)

def stochasticGradient():
    # declaring global variables
    global train_w
    global train_x
    global train_y
    global test_x
    global test_y

    # initializing variables
    N = 10000
    batch = 1000
    alpha = 0.000001
    epoch = 0 # number of iterations
    prediction = 0

    while (True):
        epoch += 1
        prediction = 0

        '''
        np.random.shuffle(train_x)
        np.random.shuffle(train_y)
        # calculating batch data
        batch_train_x = train_x[0:batch]
        batch_train_y = train_y[0:batch]
        '''

        batch_train_x, batch_train_y = unison_shuffled_copies(train_x, train_y)
        # numerator of the gradient function
        resize_y = np.transpose([batch_train_y, ]*780)
        gradient_yx = resize_y*batch_train_x

        yxw = batch_train_y * np.matmul(batch_train_x, np.transpose(train_w))
        min_function = np.sum( np.log( 1 + np.exp(-yxw) ) ) / batch # not sure

        # computer gradient
        resize_yxw = np.transpose([yxw, ]*780)
        gradient = np.sum( np.divide(-gradient_yx, (1+np.exp(resize_yxw))), axis=0 ) / batch

        train_w -= alpha * gradient

        for b in range(0, batch):
            if (np.sign(np.matmul(batch_train_x[b], np.transpose(train_w))) == batch_train_y[b]):
                prediction += 1
            else:
                prediction += -1

        train_accuracy = prediction / float(batch) * 100

        print 'Epoch:', epoch, '|| Min Function:', min_function, '|| Accuracy:', train_accuracy, '|| Gradient:', np.sum(gradient)

        if (train_accuracy > 99.95): # takes about 2000 epochs
            break

    test_w = train_w # passing in w for test...easier to follow
    testDataAccuracy(test_w, test_x, test_y, alpha)

#gradientDescent()

stochasticGradient()
