from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'


def bone_image(path):
    img_path = 'C:/Users/sudhakar/Desktop/BFD_MAJOR_PROJECT/static/uploads/image1.png'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(x)
    model = load_model('C:/Users/sudhakar/Desktop/BFD_MAJOR_PROJECT/Model.h5')
    preds = model.predict(x)
    return preds
