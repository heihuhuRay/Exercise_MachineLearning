from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

# Load the data set
data = np.loadtxt('linear.data')
#print(data)
# Separate the input from the output
X = data[:, 0:-1]
Y = data[:, -1]
N, d = X.shape
#print(X.shape)

# Separate the positive from the negative class
positive_class = X[Y==1., :]
#print(positive_class)
negative_class = X[Y==-1., :]

def Visual(x,y):

	plt.plot(positive_class[:,0], positive_class[:,1], '.r')
	plt.plot(negative_class[:,0], negative_class[:,1], '.b')
	if w[f_W] != 0:
		x = np.linspace(0, 1.0, 100)
		y = -(w[0]*x + b)/w[f_W]
		plt.plot(x,y)
	plt.show()

def Calculate_err(w,b):
	Nof_miss = 0.0
	gamma = 1000.	#just use a large num to assign gamma
	for i in range(N):
		Pr = np.dot(X[i,:],w)+b
		gamma = min(gamma, Y[i]*Pr)
		if np.sign(Pr) != Y[i]: #responsible of percetron
			Nof_miss += 1
	return Nof_miss/N, gamma	#value has to be reutrned outside the loop

"""def Visual(w,b):
	plt.plot(w[:,0], w[:,1], '.r')
	#plt.plot(negative_class[:,0], negative_class[:,1], '.b')
	plt.plot(b,'.b')
	plt.show()"""


eta = 0.1
#Initialize the weight vector and bias to 0
w = np.zeros(d)
b = 0.0
for i in range(N):
	f_W = np.sign(np.dot(X[i,:],w)+b)
	w = w + eta*(Y[i] - f_W)*X[i,:] #Y[i] just means one single value, while X[i,:] means the whole column
	b = b + eta*(Y[i] - f_W)		# the rules for bais(Online Version) in lecture02.page 27 
	#print(w, b)
Visual(w,b)
'''
print(Calculate_err(w,b))
Err, gamma = Calculate_err(w,b)
print('Functional margin:', gamma)

if gamma < 0:
	while True:
		Calculate_err(w, b)
	if np.linalg.norm(w - old_w) < 0.01*np.linalg.norm(old_w):
		pass
'''