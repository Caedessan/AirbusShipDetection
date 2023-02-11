import numpy as np
import skimage
from PIL import Image
from matplotlib.image import imread
import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, MaxPooling2D, Dropout, concatenate, Activation, Input
from keras.models import Model
from keras import backend as K
from typing import Callable
import keras
from keras import layers
''''''
def applyMask(picPath, mask):
    '''Function for applying a mask to a picture'''
    try:
        foo = Image.open(picPath)

        foo = foo.resize((256, 256), Image.Resampling.LANCZOS)
        foo = foo.convert('RGB')
        pic = np.array(foo)
    except:
        pic = imread("data/test_v2/00c3db267.jpg")
    picArray = np.array(pic)
    for i in range(picArray.shape[0]):
        for j in range(picArray.shape[1]):
            if mask[i][j] == 1:
                picArray[i][j][0] = 255
    return picArray

def saveOutput(outputPath, pic):
    '''Function designated to save networks output'''
    img = Image.fromarray(pic)
    img.save(outputPath)

def encode(image):
    flattened = image.flatten()
    encoded = ""
    i = 0
    while i < 256*256:
        if flattened[i] == 1:
            j = i
            while j< 256*256:
                if flattened[j] == 0:
                    break
                j+=1
            encoded+=" "+str(i)+" "+str(j-i)
        i+=1
    return encoded

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def decode(df):
    '''Function for decoding mask data'''
    image = np.zeros((768*768))
    amountOfRows = df.shape[0]
    if amountOfRows >= 2:
        l = []
        encodedImage = np.array(l)
        for row in range(amountOfRows):
            encodedImage = np.append(encodedImage, str(df["EncodedPixels"][row]).split())
    else:
        encodedImage = np.array(str(df["EncodedPixels"]).split())
    values = encodedImage[::2]
    amounts = encodedImage[1::2].astype(int)
    if len(amounts) == 0:
        return np.zeros((256,256))
    while True:
        for i in range(int(amounts[0])):
            image[int(values[0])+i-1]=1
        values = values[1:]
        amounts = amounts[1:]
        if len(amounts) == 0:
            break
    return skimage.measure.block_reduce(image.reshape((768,768)), (3,3), np.max).T

'''Loss function'''


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    JL = -jacard_coef(y_true, y_pred)
    return JL

def conv2d_block(input_tensor, n_filters, kernel_size=3):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def get_unet(input_img, n_filters=16, dropout=0.1):
    '''Function for getting model structure'''
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size=3)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

