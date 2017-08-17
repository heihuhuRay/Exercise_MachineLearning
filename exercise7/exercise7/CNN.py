'''
    Trains a simple CNN on the MNIST dataset.

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
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.regularizers import l2, l1
from keras.callbacks import History
from keras.utils import np_utils

# Definition of the model
def create_cnn():
    model = Sequential()

    # First convolutional layer
    model.add(Convolution2D(16, 3, 3, border_mode='valid', input_shape=(1, 28, 28)))
    model.add(Activation('relu'))

    # Max-pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flatten the convolutional layers
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # Softmax output layer
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # Learning rule
    optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    # Loss function
    model.compile(
        loss='categorical_crossentropy', # loss
        optimizer=optimizer, # learning rule
        metrics=['accuracy'] # show accuracy
    )

    return model

def load_data():
    "Loads and normalizes the training data."

    # Load the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape and normalize the data
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32') / 255.
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32') / 255.
    X_mean = np.mean(X_train, axis=0)
    X_train -= X_mean
    X_test -= X_mean
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    return (X_train, Y_train), (X_test, Y_test), X_mean


if __name__=='__main__':

    # Create the network
    model = create_cnn()

    # Print a summary of the network
    model.summary()

    # Load the data
    (X_train, Y_train), (X_test, Y_test), X_mean = load_data()

    # Train for 20 epochs using minibatches
    history = History()
    try:
        model.fit(X_train, Y_train,
            batch_size=200, 
            nb_epoch=20,
            validation_split=0.1,
            callbacks=[history])
        
    except KeyboardInterrupt:
        pass

    # Compute the test accuracy
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('\nTest loss:', score[0])
    print('Test accuracy:', score[1])

    # Save the model
    model.save('cnn-mnist.h5')

    # Show accuracy
    plt.plot(history.history['acc'], '-r', label="Training")
    plt.plot(history.history['val_acc'], '-b', label="Validation")
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

