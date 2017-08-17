from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
 
# Load the data set
data = np.loadtxt('polynome.data')
X = data[:, 0]
Y = data[:, 1]
N = len(X)

def visualize(w):
    # Plot the data
    plt.plot(X, Y, 'r.')
    # Plot the fitted curve 
    x = np.linspace(0., 1., 100)
    y = np.polyval(w, x)
    plt.plot(x, y, 'g-')
    plt.title('Polynomial regression with order ' + str(len(w)-1))
    plt.show()

# Apply polynomial regression of order 2 on the data
#w = np.polyfit(X, Y, 3)

sum_Quadratic_Error = 0
for i in range(20):
	w = np.polyfit(X, Y, i)
	#pylab.ylim([-1,2])
	y_calculated = np.polyval(w, X)
	Quadratic_Error = 0.5*((Y - y_calculated)**2)
	print(Quadratic_Error)
	sum_Quadratic_Error = Quadratic_Error + sum_Quadratic_Error
	#print("quadratic error for Num",i,"is",sum_Quadratic_Error)
	#visualize(w) # Visualize the fit #visualize(w)


'''
for deg in range(1,20):
	w = np.polyfit(X, Y, deg)
	y_fit = np.polyval(w, X)
	error = 0.5*np.dot((Y - y_fit).T,(Y-y_fit)) # T means transform, while Y is a vector
	print("Order =",deg,"Error = ",error)
'''
limit = int(0.7*N)
X_train, X_test = np.hsplit(X, [limit])
Y_train, Y_test = np.hsplit(Y, [limit])
for deg in range(1,11):
	w = np.polyfit(X_train, Y_train, deg)
	y_fit_train = np.polyval(w, X_train)
	y_fit_test = np.polyval(w, X_test)
	train_err = 0.5*np.dot((Y_train - y_fit_train).T,(Y_train - y_fit_train))
	test_err = 0.5*np.dot((Y_test - y_fit_test).T,(Y_test - y_fit_test))
	print('Order =',deg,'train_err =',train_err,'test_err =',test_err)
	