from cnn_architecture import model_architecture
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import random
import cv2
import pickle
import os


## Function to load images
def load_data(path, image_size):
    data = []
    labels = []
    image_paths = sorted(list(paths.list_images(path)))
    random.seed(7)
    random.shuffle(image_paths)
    for i in image_paths:
        image = cv2.imread(i)
        image = cv2.resize(image, (image_size[0], image_size[1]))
        image = img_to_array(image)
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


## Saving the model
def saving_models_and_labels(model, lb, model_name, label_name):
    model.save(os.getcwd() + '/CNN_models/{}'.format(model_name), save_format='h5')
    f = open(os.getcwd() + '/CNN_models/{}'.format(label_name), 'wb')
    f.write(pickle.dumps(lb))
    f.close()


## Fitting the model
def model_fitting(model, xtrain, xtest, ytrain, ytest, epochs, batch_size):
    model.fit(x=aug.flow(xtrain, ytrain, batch_size=batch_size), validation_data=(xtest, ytest), epochs=epochs)
    ypred = model.predict(xtest, batch_size=batch_size)
    print(classification_report(ytest.argmax(axis=1), ypred.argmax(axis=1), target_names=lb.classes_))
    return model


## Model Definition
def model_definition(model, lr, epochs):
    # decay is actually the learning rate decay
    opt = Adam(learning_rate=lr, decay=lr / epochs)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    epochs = 100
    lr = 1e-3
    batch_size = 32
    image_dims = (96, 96, 3)
    pwd = os.getcwd()
    data, labels = load_data('pokemon', image_dims)
    lb = LabelBinarizer()
    xtrain, xtest, ytrain, ytest = splitting(data, labels)
    ytrain = lb.fit_transform(ytrain)
    ytest = lb.transform(ytest)
    # generating more images - augmentation
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")
    model = model_architecture(image_dims[1], image_dims[0], image_dims[2], classes=len(lb.classes_))
    model = model_definition(model, lr, epochs)
    model = model_fitting(model, xtrain, xtest, ytrain, ytest, epochs, batch_size)
    saving_models_and_labels(model, lb, 'first_model', 'classes')
