import numpy as np
from math import exp
import torch

# load training data
#x = torch.from_numpy(np.loadtxt('mnist_X_train.dat'))
x = np.loadtxt('mnist_X_train.dat')
y = np.loadtxt('mnist_y_train.dat')
#w = torch.DoubleTensor(780,1).uniform_(-0.01, 0.01)
#w = -0.01-0.01 * torch.rand(780,1) + 0.01
w = np.random.uniform(-0.01, 0.01, 780)

N = 10000
alpha = 0.00001

while (True):
    #xw = np.matmul(x,np.transpose(w))
    yxw = y * np.matmul(x,np.transpose(w))
    min_function = np.sum( np.log( 1 + np.exp(-yxw) ) ) / N

    # computer gradient
    resize_y = np.transpose([y, ]*780)
    gradient_yx = resize_y*x
    #gradient_xw = np.matmul(x, np.transpose(w))
    #gradient_yxw = (y * gradient_xw)
    resize_yxw = np.transpose([yxw, ]*780)
    gradient = np.sum( np.divide(-gradient_yx, (1+np.exp(resize_yxw))), axis=0 ) / N

    w -= alpha * gradient
    print 'Function: ', min_function, '  Gradient: ', np.sum(gradient)

    if (np.absolute(np.sum(gradient)) < 0.001):
        break
