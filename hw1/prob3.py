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
    print '\n'
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
    alpha = 0.00001 # 0.00001 best alpha; should not go below 0.000001
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

        train_w -= alpha * gradient

        for n in range(0, N):
            if (np.sign(np.matmul(train_x[n], np.transpose(train_w))) == train_y[n]):
                prediction += 1
            else:
                prediction += -1

        train_accuracy = prediction / float(N) * 100

        norm_gradient = np.linalg.norm(gradient)
        print 'Epoch:', epoch, '|| Min Function:', min_function, '|| Accuracy:', train_accuracy, '|| Gradient:', np.sum(gradient), '|| Norm:', norm_gradient

        if (train_accuracy > 99.0 and norm_gradient < 1): # takes about 2000 epochs
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
    alpha = 0.0000001
    epoch = 0 # number of iterations
    prediction = 0
    vary = False

    while (True):
        epoch += 1
        prediction = 0

        batch_train_x, batch_train_y = unison_shuffled_copies(train_x, train_y)
        batch_train_x = batch_train_x[0:batch]
        batch_train_y = batch_train_y[0:batch]
        #print 'X:', np.shape(batch_train_x)
        #print 'Y:', np.shape(batch_train_y)
        #time.sleep(10)

        # numerator of the gradient function
        resize_y = np.transpose([batch_train_y, ]*780)
        gradient_yx = resize_y*batch_train_x

        yxw = batch_train_y * np.matmul(batch_train_x, np.transpose(train_w))
        min_function = np.sum( np.log( 1 + np.exp(-yxw) ) ) / batch # not sure

        # computer gradient
        resize_yxw = np.transpose([yxw, ]*780)
        gradient = np.sum( np.divide(-gradient_yx, (1+np.exp(resize_yxw))), axis=0 ) / batch
        # Problem 3d
        # conditions to vary the step size
        if (vary):
            alpha = np.power(alpha, 1.0002)

        #if (alpha < 0.000001 and vary):
        #    alpha = 0.000001

        #print('%.15f' % alpha)

        train_w -= alpha * gradient

        for b in range(0, batch):
            if (np.sign(np.matmul(batch_train_x[b], np.transpose(train_w))) == batch_train_y[b]):
                prediction += 1
            else:
                prediction += -1

        train_accuracy = prediction / float(batch) * 100
        norm_gradient = np.linalg.norm(gradient)
        print 'Epoch:', epoch, '|| Min Function:', min_function, '|| Accuracy:', train_accuracy, '|| Gradient:', np.sum(gradient), '|| Norm:', norm_gradient

        if (train_accuracy > 98.0): # takes about 2000 epochs
            break

    test_w = train_w # passing in w for test...easier to follow
    testDataAccuracy(test_w, test_x, test_y, alpha)

def LSVM():
    # declaring global variables
    global train_w
    global train_x
    global train_y
    global test_x
    global test_y

    N = 10000
    batch = 1000
    alpha = 0.00001
    epoch = 0 # number of iterations
    prediction = 0
    vary = True

    while (True):
        epoch += 1
        prediction = 0

        batch_train_x, batch_train_y = unison_shuffled_copies(train_x, train_y)
        batch_train_x = batch_train_x[0:batch]
        batch_train_y = batch_train_y[0:batch]

        hinge_loss = 0
        for b in range(0, batch):
            one_row_yxw = batch_train_y[b] * np.matmul(batch_train_x[b], np.transpose(train_w))
            if (one_row_yxw < 1):
                hinge_loss += 1-one_row_yxw
            elif (one_row_yxw > 1):
                hinge_loss += 0
        hinge_loss /= batch

        gradient = 0
        for b in range(0, batch):
            one_row_yxw = batch_train_y[b] * np.matmul(batch_train_x[b], np.transpose(train_w))
            if (one_row_yxw < 1):
                gradient += (-batch_train_y[b] * batch_train_x[b])
            elif (one_row_yxw > 1):
                gradient += 0
        gradient /= batch

        if (vary):
            alpha = np.power(alpha, 1.00015)

        #if (alpha < 0.000000001 and vary): # needs to be super low for this problem
        #    alpha = 0.000000001
        print('Alpha: %.20f' % alpha)

        train_w -= alpha * gradient

        for b in range(0, batch):
            if (np.sign(np.matmul(batch_train_x[b], np.transpose(train_w))) == batch_train_y[b]):
                prediction += 1
            else:
                prediction += -1

        train_accuracy = prediction / float(batch) * 100

        norm_gradient = np.linalg.norm(gradient)
        print 'Epoch:', epoch, '|| Min Function:', hinge_loss, '|| Accuracy:', train_accuracy, '|| Gradient:', np.sum(gradient), '|| Norm:', norm_gradient

        if (train_accuracy > 98.0 and norm_gradient < 10 and epoch > 5000): # takes about 2000 epochs
            break

    test_w = train_w # passing in w for test...easier to follow
    testDataAccuracy(test_w, test_x, test_y, alpha)


#gradientDescent()

stochasticGradient()

#LSVM()
