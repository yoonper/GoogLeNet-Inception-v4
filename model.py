#!/usr/bin/python
# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Activation, Input, concatenate, Dropout, Dense, Flatten
from keras.layers import MaxPooling2D, Conv2D, AveragePooling2D, BatchNormalization


def create(classes_num=1000, image_height=299, image_width=299, image_channel=3):
    inputs = Input((image_height, image_width, image_channel))

    # 299 x 299 x 3
    net = block_stem(inputs)

    # 4 x Inception-A ( Output: 35 x 35 x 384 )
    for i in range(4):
        net = block_inception_a(net)

    # Reduction-A ( Output: 17 x 17 x 1024 )
    net = block_reduction_a(net)

    # 7 x Inception-B ( Output: 17 x 17 x 1024 )
    for i in range(7):
        net = block_inception_b(net)

    # Reduction-B ( Output: 8 x 8 x 1536 )
    net = block_reduction_b(net)

    # 3 x Inception-C ( Output: 8 x 8 x 1536 )
    for i in range(3):
        net = block_inception_c(net)

    # Average Pooling ( Output: 1536 )
    net = AveragePooling2D((8, 8))(net)

    # Dropout ( keep 0.8 )
    net = Dropout(0.2)(net)
    net = Flatten()(net)

    # Output
    outputs = Dense(units=classes_num, activation='softmax')(net)

    return Model(inputs, outputs, name='Inception-v4')


def block_stem(inputs):
    net = conv2d(inputs, 32, (3, 3), strides=(2, 2), padding='valid')
    net = conv2d(net, 32, (3, 3), padding='valid')
    net = conv2d(net, 64, (3, 3))

    branch_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(net)
    branch_2 = conv2d(net, 96, (3, 3), strides=(2, 2), padding='valid')

    net = concatenate([branch_1, branch_2])

    branch_1 = conv2d(net, 64, (1, 1))
    branch_1 = conv2d(branch_1, 96, (3, 3), padding='valid')

    branch_2 = conv2d(net, 64, (1, 1))
    branch_2 = conv2d(branch_2, 64, (7, 1))
    branch_2 = conv2d(branch_2, 64, (1, 7))
    branch_2 = conv2d(branch_2, 96, (3, 3), padding='valid')

    net = concatenate([branch_1, branch_2])

    branch_1 = conv2d(net, 192, (3, 3), strides=(2, 2), padding='valid')  # different from the paper
    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(net)

    net = concatenate([branch_1, branch_2])

    return net


def block_inception_a(inputs):
    branch_1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch_1 = conv2d(branch_1, 96, (1, 1))

    branch_2 = conv2d(inputs, 96, (1, 1))

    branch_3 = conv2d(inputs, 64, (1, 1))
    branch_3 = conv2d(branch_3, 96, (3, 3))

    branch_4 = conv2d(inputs, 64, (1, 1))
    branch_4 = conv2d(branch_4, 96, (3, 3))
    branch_4 = conv2d(branch_4, 96, (3, 3))

    return concatenate([branch_1, branch_2, branch_3, branch_4])


def block_inception_b(inputs):
    branch_1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch_1 = conv2d(branch_1, 128, (1, 1))

    branch_2 = conv2d(inputs, 384, (1, 1))

    branch_3 = conv2d(inputs, 192, (1, 1))
    branch_3 = conv2d(branch_3, 224, (1, 7))
    branch_3 = conv2d(branch_3, 256, (7, 1))  # different from the paper

    branch_4 = conv2d(inputs, 192, (1, 1))
    branch_4 = conv2d(branch_4, 192, (1, 7))
    branch_4 = conv2d(branch_4, 224, (7, 1))
    branch_4 = conv2d(branch_4, 224, (1, 7))
    branch_4 = conv2d(branch_4, 256, (7, 1))

    return concatenate([branch_1, branch_2, branch_3, branch_4])


def block_inception_c(inputs):
    branch_1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch_1 = conv2d(branch_1, 256, (1, 1))

    branch_2 = conv2d(inputs, 256, (1, 1))

    branch_3 = conv2d(inputs, 384, (1, 1))
    branch_3_1 = conv2d(branch_3, 256, (1, 3))
    branch_3_2 = conv2d(branch_3, 256, (3, 1))

    branch_4 = conv2d(inputs, 384, (1, 1))
    branch_4 = conv2d(branch_4, 448, (1, 3))
    branch_4 = conv2d(branch_4, 512, (3, 1))
    branch_4_1 = conv2d(branch_4, 256, (3, 1))
    branch_4_2 = conv2d(branch_4, 256, (1, 3))

    return concatenate([branch_1, branch_2, branch_3_1, branch_3_2, branch_4_1, branch_4_2])


def block_reduction_a(inputs):
    branch_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(inputs)

    branch_2 = conv2d(inputs, 384, (3, 3), strides=(2, 2), padding='valid')

    branch_3 = conv2d(inputs, 192, (1, 1))
    branch_3 = conv2d(branch_3, 224, (3, 3))
    branch_3 = conv2d(branch_3, 256, (3, 3), strides=(2, 2), padding='valid')

    return concatenate([branch_1, branch_2, branch_3])


def block_reduction_b(inputs):
    branch_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(inputs)

    branch_2 = conv2d(inputs, 192, (1, 1))
    branch_2 = conv2d(branch_2, 192, (3, 3), strides=(2, 2), padding='valid')

    branch_3 = conv2d(inputs, 256, (1, 1))
    branch_3 = conv2d(branch_3, 256, (1, 7))
    branch_3 = conv2d(branch_3, 320, (7, 1))
    branch_3 = conv2d(branch_3, 320, (3, 3), strides=(2, 2), padding='valid')

    return concatenate([branch_1, branch_2, branch_3])


def conv2d(net, filters, kernel_size, strides=(1, 1), padding='same'):
    net = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    return net


if __name__ == '__main__':
    model = create()
    model.summary()