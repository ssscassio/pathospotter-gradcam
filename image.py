from PIL import Image
import numpy as np
import scipy.misc
import cv2

def load_image(image_path):
    image = Image.open(image_path, ) # Open Image
    image = image.convert('RGB') # Convert to RGB
    image = np.asarray(image) # Convert to numpy array
    return image

'''
Preprocess Image resizing it to model's firts layer shape
and scale rgb values to [0,1] 
'''
def preprocess_image(image, size=(224, 224)):
    _, _, chan = image.shape
    assert chan == 3
    image = scipy.misc.imresize(image, size, interp='bilinear')
    image = image/255.
    image = np.array(image, dtype='float32')
    image = np.expand_dims(image, axis=0)
    return image

def save_image(image, file_path):
    cv2.imwrite(file_path, image)
