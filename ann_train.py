from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from imutils import paths
import numpy as np
import pickle
import cv2
import os


## Overview:
# This code is for training a basic ANN model with 3 Dense(FC) layers.
# Input---Hidden(1024)---Hidden(512)---Output
# Images have been resized to 32x32
# Test size has been hardcoded to 0.3
# Batch size- 32; 70 epochs; lr=0.01

## Function to load images
def load_data(path):
    data = []
    labels = []
    image_paths = sorted(list(paths.list_images(path)))
    for i in image_paths:
        image = cv2.imread(i)
        image = cv2.resize(image, (32, 32)).flatten()
        data.append(image)
        label = i.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    return data, labels


## Splitting data
def splitting(data, labels):
    xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.3, random_state=7)
    return xtrain, xtest, ytrain, ytest


## Defining the ANN Model
def model_definition(learning_rate):
    model = Sequential()
    model.add(Dense(1024, input_shape=(3072,), activation='sigmoid'))
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dense(len(lb.classes_), activation='softmax'))
    print('Model Created... Training it now...')
    opt = SGD(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    # for binary classification the loss would have been "binary_crossentropy"
    return model


## Fitting the model
def model_fitting(model, xtrain, xtest, ytrain, ytest, epochs, batch_size):
    model.fit(x=xtrain, y=ytrain, validation_data=(xtest, ytest), epochs=epochs, batch_size=batch_size)
    ypred = model.predict(xtest, batch_size=batch_size)
    print(classification_report(ytest.argmax(axis=1), ypred.argmax(axis=1), target_names=lb.classes_))
    return model


## Saving the model
def saving_models_and_labels(model, lb, model_name, label_name):
    model.save(os.getcwd() + '/models/{}'.format(model_name), save_format='h5')
    f = open(os.getcwd() + '/models/{}'.format(label_name), 'wb')
    f.write(pickle.dumps(lb))
    f.close()


if __name__ == '__main__':
    lb = LabelBinarizer()
    data, labels = load_data('animals')  # folder for data
    xtrain, xtest, ytrain, ytest = splitting(data, labels)
    ytrain = lb.fit_transform(ytrain)
    ytest = lb.transform(ytest)
    model = model_definition(learning_rate=0.01)  # change this
    model = model_fitting(model, xtrain, xtest, ytrain, ytest, epochs=70, batch_size=32)  # change this
    saving_models_and_labels(model, lb, 'first_model', 'classes')  # can change these names
