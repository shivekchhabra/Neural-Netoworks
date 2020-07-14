import pickle
import cv2
import os
from tensorflow.keras.models import load_model


## General code for loading model and labels.
def loading_model(path_model, path_labels):
    model = load_model(path_model)
    lb = pickle.loads(open(path_labels, 'rb').read())
    return model, lb


## For basic image processing (as done while training)
def same_processing_as_model(image):
    image = cv2.resize(image, (32, 32))
    image = image.astype("float") / 255.0
    image = image.flatten()
    image = image.reshape((1, image.shape[0]))
    return image


if __name__ == '__main__':
    pwd = os.getcwd()
    model, lb = loading_model(pwd + '/models/first_model', pwd + '/models/classes')
    image = cv2.imread(pwd + '/Samples/dog.jpeg')
    output = image.copy()
    image = same_processing_as_model(image)
    preds = model.predict(image)
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]
    text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
    cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2)
    cv2.imshow("Image", output)
    cv2.waitKey(0)
