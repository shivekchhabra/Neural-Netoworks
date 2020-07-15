import pickle
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


## General code for loading model and labels.
def loading_model(path_model, path_labels):
    model = load_model(path_model)
    lb = pickle.loads(open(path_labels, 'rb').read())
    return model, lb


## For basic image processing (as done while training)
def same_processing_as_model(image):
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # to add a new dimension ( from (2,) to (1,2,)
    return image


if __name__ == '__main__':
    pwd = os.getcwd()
    model, lb = loading_model(pwd + '/CNN_Models/first_model', pwd + '/CNN_Models/classes')
    image = cv2.imread(pwd + '/CNN_Samples/pikachu.png')
    output = image.copy()
    image = same_processing_as_model(image)
    preds = model.predict(image)
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]
    text = "{}".format(label)
    cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2)
    cv2.imshow("Image", output)
    cv2.waitKey(0)