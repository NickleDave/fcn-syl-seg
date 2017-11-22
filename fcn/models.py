from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Lambda
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.utils import to_categorical

from keras_fcn.encoders import Encoder
from keras_fcn.decoders import VGGUpsampler
from keras_fcn.blocks import (vgg_conv, vgg_fc)


def fw_conv1():
    """1st convolution layer for flatwindow
    """
    def f(x):
        x = Conv2D(16, (5, 5), activation='relu')(x)
        x = Conv2D(16, (1, 1), activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        return x
    return f


def fw_conv2():
    """2nd convolution layer for flatwindow
    """
    def f(x):
        x = Conv2D(16, (4, 4), activation='relu')(x)
        x = Conv2D(16, (1, 1), activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        return x
    return f


def fcn_custom(fcn_custom_input_shape, weights_filename):
    inputs = Input(shape=fcn_custom_input_shape)
    blocks = [fw_conv1(),
              fw_conv1(),
              fw_conv2(),
              vgg_fc(fw_for_fcn.layers[-4].get_config()['units'])]
    encoder = Encoder(inputs, blocks, weights=weights_filename,
                      trainable=True)
    feat_pyramid = encoder.outputs  # A feature pyramid with 5 scales
    # feat_pyramid = feat_pyramid[:3]  # Select only the top three scale of the pyramid
    feat_pyramid.append(inputs)  # Add image to the bottom of the pyramid

    outputs = VGGUpsampler(feat_pyramid, scales=[1, 1e-2, 1e-4], classes=num_classes)
    outputs = Activation('softmax')(outputs)

    fcn_custom = Model(inputs=inputs, outputs=outputs)
    return fcn_custom