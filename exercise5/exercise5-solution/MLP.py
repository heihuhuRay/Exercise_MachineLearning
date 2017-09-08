'''
    Trains a simple MLP on the MNIST dataset.

    Adapted from the Keras example:

    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
'''
# Usual imports
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

# Import Keras objects
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.regularizers import l2, l1
from keras.callbacks import History
from keras.utils import np_utils

# Number of classes
nb_classes = 10

# Load the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize the data
X_train = X_train.reshape(X_train.shape[0], 784).astype('float32') / 255.
X_test = X_test.reshape(X_test.shape[0], 784).astype('float32') / 255.
X_mean = np.mean(X_train, axis=0)
X_train -= X_mean
X_test -= X_mean
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Definition of the model
def create_mlp():
    model = Sequential()

    # Hidden layer with 100 neurons, taking inputs from the 784 pixels 
    model.add(Dense(100, input_shape=(784,))) # Weights
    model.add(Activation('relu')) # Transfer function
    model.add(BatchNormalization()) # Batch normalization
    model.add(Dropout(0.2)) # Dropout

    # Second hidden layer
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Softmax output layer
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # Learning rule
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # or:
    # optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    # Loss function
    model.compile(
        loss='categorical_crossentropy', # loss
        optimizer=optimizer, # learning rule
        metrics=['accuracy'] # show accuracy
    )

    return model

# Create the model
model = create_mlp()

# Print a summary of the network
model.summary()

# Train for 20 epochs using minibatches
history = History()
try:
    model.fit(X_train, Y_train,
        batch_size=50, 
        nb_epoch=20,
        validation_split=0.1,
        callbacks=[history])
    
except KeyboardInterrupt:
    pass

# Compute the test accuracy
score = model.evaluate(X_test, Y_test, verbose=0)
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])

# Show accuracy
plt.figure(1)
plt.plot(history.history['acc'], '-r', label="Training")
plt.plot(history.history['val_acc'], '-b', label="Validation")
plt.xlabel('Epoch #')
plt.ylabel('Accuracy')
plt.legend()

# Show misclassified examples
Y_pred = model.predict_classes(X_test, verbose=0)
misclassification = (Y_pred != y_test).nonzero()[0]
plt.figure(2)
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.imshow((X_test[misclassification[i], :] + X_mean).reshape((28, 28)))
    plt.title('Prediction: '+ str(Y_pred[misclassification[i]]) + '; Target: ' + str(y_test[misclassification[i]]))
plt.show()
