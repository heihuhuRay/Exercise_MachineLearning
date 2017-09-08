from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def hypothesis(input, _Y_):
    _h_ = sigmoid(input)
    out = (_h_**(1-_Y_))*((1-_h_)**_Y_)
    return	out

def log_loss(_h_, _Y_):
  	loss = -_Y_*np.log(_h_) - (1-_Y_)*np.log(1-_h_)  

# Load the data set
data = np.loadtxt('linear.data')

# Separate X from Y
X = data[:, 0:-1]
Y = data[:, -1]
N, d = X.shape# N:100, d:2
print(X)
w = np.zeros(d)# init w:0
b = 0 #init b:0
a = 0.1 #learning rate

Input_X = np.dot(X,w)+b

for i in range(N):
	y = Y[i]
	h = hypothesis(Input_X[i], y)
	loss = log_loss(h, y)
	w = w - a*
