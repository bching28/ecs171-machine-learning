import numpy as np
from math import exp
import time
#import torch

# load training data
#x = torch.from_numpy(np.loadtxt('mnist_X_train.dat'))
x = np.loadtxt('mnist_X_train.dat')
y = np.loadtxt('mnist_y_train.dat')
#w = torch.DoubleTensor(780,1).uniform_(-0.01, 0.01)
#w = -0.01-0.01 * torch.rand(780,1) + 0.01
w = np.random.uniform(-0.01, 0.01, 780)

N = 10000
alpha = 0.0001
n_epoch = 100
'''
for epoch in range(n_epoch):
    for n in range(1, N):
        temp = (y[n]*x[n]) / (1+np.exp(y[n]*(np.transpose(w)*x[n])))

    gradient = (-1/N) * temp
    print 'Gradient:', np.sum(gradient)

    w -= alpha * gradient

for n in range(1, N):
    temp = np.log(1 + np.exp(-y[n]*(np.transpose(w)*x[n])))

min_function = (1/N)*temp
print np.shape(min_function)
print 'Min Function:', min_function
'''

prediction = 0
resize_y = np.transpose([y, ]*780)
gradient_yx = resize_y*x
epoch = 0

while (True):
#for epoch in range(n_epoch):
    epoch += 1
    prediction = 0
    yxw = y * np.matmul(x,np.transpose(w))
    min_function = np.sum( np.log( 1 + np.exp(-yxw) ) ) / N

    # computer gradient
    resize_yxw = np.transpose([yxw, ]*780)
    gradient = np.sum( np.divide(-gradient_yx, (1+np.exp(resize_yxw))), axis=0 ) / N

    w -= alpha * gradient
    #sig = np.sign(np.transpose(w) * x[n])
    #for n in range(0, N):
    #    print np.shape(np.sign(np.matmul(x[n],np.transpose(w))))
    #    time.sleep(3)

    for n in range(0, N):

        if (np.sign(np.matmul(x[n],np.transpose(w))) == y[n]):
            prediction += 1
        else:
            prediction += -1

    #print prediction
    accuracy = prediction / float(N) * 100

    print 'Epoch:', epoch, '|| Min Function:', min_function, '|| Accuracy:', accuracy, '|| Gradient:', np.sum(gradient)
    #print 'Gradient:', np.sum(gradient)

    #if (np.absolute(np.sum(gradient)) < 0.001):
    #    break

    if (accuracy > 99.0): # takes about 2000 epochs
        break

#min_function = np.sum( np.log( 1 + np.exp(-yxw) ) ) / N
#print 'Min Function:', min_function
#print 'Weights:', w
