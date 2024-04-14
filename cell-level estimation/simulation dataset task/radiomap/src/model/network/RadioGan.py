import os
# import tensorflow as tf
import numpy as np
# from tensorflow import keras
import torch
import torch.nn as nn

# class Downsample(keras.Model):
#
#     def __init__(self, filters, size, apply_batchnorm=True):
#         super(Downsample, self).__init__()
#
#         self.apply_batchnorm = apply_batchnorm
#         initializer = tf.random_normal_initializer(0., 0.02)
#         # self.conv0 = keras.layers.Conv2D(filters,
#         #                                  (size, size),
#         #                                  strides=1,
#         #                                  padding='same',
#         #                                  kernel_initializer=initializer,
#         #                                  use_bias=False)
#         self.conv1 = keras.layers.Conv2D(filters,
#                                          (size, size),
#                                          strides=2,
#                                          padding='same',
#                                          kernel_initializer=initializer,
#                                          use_bias=False)
#         if self.apply_batchnorm:
#             self.batchnorm = keras.layers.BatchNormalization()
#
#     def call(self, x, training):
#         # x = self.conv0(x)
#         x = self.conv1(x)
#         if self.apply_batchnorm:
#             x = self.batchnorm(x, training=training)
#         x = tf.nn.leaky_relu(x)
#         return x

class Downsample(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, padding, apply_batchnorm=True):
        super(Downsample, self).__init__()

        self.apply_batchnorm = apply_batchnorm
        # initializer = tf.random_normal_initializer(0., 0.02)
        # self.conv0 = keras.layers.Conv2D(filters,
        #                                  (size, size),
        #                                  strides=1,
        #                                  padding='same',
        #                                  kernel_initializer=initializer,
        #                                  use_bias=False)
        # self.conv1 = keras.layers.Conv2D(filters,
        #                                  (size, size),
        #                                  strides=2,
        #                                  padding='same',
        #                                  kernel_initializer=initializer,
        #                                  use_bias=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=2, padding=padding)
        self.leaky_relu = nn.LeakyReLU()
        if self.apply_batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x = self.conv0(x)
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x)
        # x = tf.nn.leaky_relu(x)
        x = self.leaky_relu(x)
        return x

# class Upsample(keras.Model):
#
#     def __init__(self, filters, size, apply_dropout=True):
#         super(Upsample, self).__init__()
#
#         self.apply_dropout = apply_dropout
#         initializer = tf.random_normal_initializer(0., 0.02)
#         # self.up_conv_0 = keras.layers.Conv2DTranspose(filters,
#         #                                             (size, size),
#         #                                             strides=1,
#         #                                             padding='same',
#         #                                             kernel_initializer=initializer,
#         #                                             use_bias=False)
#         self.up_conv = keras.layers.Conv2DTranspose(filters,
#                                                     (size, size),
#                                                     strides=2,
#                                                     padding='same',
#                                                     kernel_initializer=initializer,
#                                                     use_bias=False)
#         self.batchnorm = keras.layers.BatchNormalization()
#         if self.apply_dropout:
#             self.dropout = keras.layers.Dropout(0.5)
#
#     def call(self, x1, x2, training=None):
#         # x = self.up_conv_0(x1)
#         x = self.up_conv(x1)
#         x = self.batchnorm(x, training=training)
#         if self.apply_dropout:
#             x = self.dropout(x, training=training)
#         x = tf.nn.relu(x)
#         x = tf.concat([x, x2], axis=-1)
#         return x

class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, padding, apply_dropout=True):
        super(Upsample, self).__init__()

        self.apply_dropout = apply_dropout
        # initializer = tf.random_normal_initializer(0., 0.02)
        # self.up_conv_0 = keras.layers.Conv2DTranspose(filters,
        #                                             (size, size),
        #                                             strides=1,
        #                                             padding='same',
        #                                             kernel_initializer=initializer,
        #                                             use_bias=False)
        # self.up_conv = keras.layers.Conv2DTranspose(filters,
        #                                             (size, size),
        #                                             strides=2,
        #                                             padding='same',
        #                                             kernel_initializer=initializer,
        #                                             use_bias=False)
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=padding)
        # self.batchnorm = keras.layers.BatchNormalization()
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        if self.apply_dropout:
            # self.dropout = keras.layers.Dropout(0.5)
            self.dropout = nn.Dropout(0.5)

    def forward(self, x1, x2):
        # x = self.up_conv_0(x1)
        x = self.up_conv(x1)
        x = self.batchnorm(x)
        if self.apply_dropout:
            x = self.dropout(x)
        # x = tf.nn.relu(x)
        x = self.relu(x)
        # x = tf.concat([x, x2], axis=1)
        x = torch.cat([x, x2], dim=1)
        return x

def convreluT(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=padding),
        nn.ReLU(inplace=True)
        #input is H X W, output is   (H-1)*2 - 2*padding + kernel
    )

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # self.down1 = Downsample(32, 4, apply_batchnorm=False)
        # self.down2 = Downsample(64, 4)
        # self.down3 = Downsample(128, 4)
        # self.down4 = Downsample(256, 4)
        # self.down5 = Downsample(512, 4)
        # self.down6 = Downsample(512, 4)

        self.down1 = Downsample(2, 32, 3, 1,  apply_batchnorm=False)
        self.down2 = Downsample(32, 64, 3, 1)
        self.down3 = Downsample(64, 128, 3, 1)
        self.down4 = Downsample(128, 256, 3, 1)
        self.down5 = Downsample(256, 512, 3, 1)
        self.down6 = Downsample(512, 512, 3, 1)

        # self.up1 = Upsample(512, 4, apply_dropout=True)
        # self.up2 = Upsample(256, 4, apply_dropout=True)
        # self.up3 = Upsample(128, 4)
        # self.up4 = Upsample(64, 4)
        # self.up5 = Upsample(32, 4)

        self.up1 = Upsample(512, 512, 4, 1, apply_dropout=True)
        self.up2 = Upsample(512 + 512, 256, 4, 1, apply_dropout=True)
        self.up3 = Upsample(256 + 256, 128, 4, 1)
        self.up4 = Upsample(128 + 128, 64, 4, 1)
        self.up5 = Upsample(64 + 64, 32, 4, 1)


        # self.last = keras.layers.Conv2DTranspose(1, (4, 4),
        #                                          strides=2,
        #                                          padding='same',
        #                                          kernel_initializer=initializer)
        self.last = convreluT(32 + 32, 1, 4, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x shape == (bs, 64, 64, 2)
        x1 = self.down1(x)  # (bs, 32, 32, 16)  (bs, 16, 16, 16)
        x2 = self.down2(x1)  # (bs, 16, 16, 32)    (bs, 8, 8, 16)
        x3 = self.down3(x2)  # (bs, 8, 8, 64)  (bs, 4, 4, 16)
        x4 = self.down4(x3)  # (bs, 4, 4, 128)  (bs, 2, 2, 16)
        x5 = self.down5(x4)  # (bs, 2, 2, 128)  (bs, 1, 1, 16)
        x6 = self.down6(x5)  # (bs, 1, 1, 128)

        u1 = self.up1(x6, x5)  # (bs, 2, 2, 128)
        u2 = self.up2(u1, x4)  # (bs, 4, 4, 128)
        u3 = self.up3(u2, x3)  # (bs, 8, 8, 64)
        u4 = self.up4(u3, x2)  # (bs, 16, 16, 32)
        u5 = self.up5(u4, x1)  # (bs, 32, 32, 16)

        # u1 = self.up1(x5, x4, training=training)  # (bs, 2, 2, 128)
        # u2 = self.up2(u1, x3, training=training)  # (bs, 4, 4, 128)
        # u3 = self.up3(u2, x2, training=training)  # (bs, 8, 8, 64)
        # u4 = self.up4(u3, x1, training=training)  # (bs, 16, 16, 32)

        out = self.last(u5)  # (bs, 64, 64, 1)
        # out = self.last(u4)  # (bs, 64, 64, 1)
        # out = tf.nn.tanh(out)
        out = self.tanh(out)

        return out


# class DiscDownsample(keras.Model):
#
#     def __init__(self, filters, size, apply_batchnorm=True):
#         super(DiscDownsample, self).__init__()
#
#         self.apply_batchnorm = apply_batchnorm
#         initializer = tf.random_normal_initializer(0., 0.02)
#
#         self.conv1 = keras.layers.Conv2D(filters, (size, size),
#                                          strides=2,
#                                          padding='same',
#                                          kernel_initializer=initializer,
#                                          use_bias=False)
#         if self.apply_batchnorm:
#             self.batchnorm = keras.layers.BatchNormalization()
#
#     def call(self, x, training=None):
#
#         x = self.conv1(x)
#         if self.apply_batchnorm:
#             x = self.batchnorm(x, training=training)
#         x = tf.nn.leaky_relu(x)
#         return x

class DiscDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding, apply_batchnorm=True):
        super(DiscDownsample, self).__init__()

        self.apply_batchnorm = apply_batchnorm
        # initializer = tf.random_normal_initializer(0., 0.02)

        # self.conv1 = keras.layers.Conv2D(filters, (size, size),
        #                                  strides=2,
        #                                  padding='same',
        #                                  kernel_initializer=initializer,
        #                                  use_bias=False)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=2, padding=padding)
        if self.apply_batchnorm:
            # self.batchnorm = keras.layers.BatchNormalization()
            self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x)
        # x = tf.nn.leaky_relu(x)
        x = self.leaky_relu(x)
        return x

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        # initializer = tf.random_normal_initializer(0., 0.02)

        # self.down1 = DiscDownsample(64, 4, False)
        # self.down2 = DiscDownsample(128, 4)
        # self.down3 = DiscDownsample(256, 4)

        self.down1 = DiscDownsample(2, 64, 3, 1, False)
        self.down2 = DiscDownsample(64, 128, 3, 1)
        self.down3 = DiscDownsample(128, 256, 3, 1)

        # we are zero padding here with 1 because we need our shape to
        # go from (batch_size, 32, 32, 256) to (batch_size, 31, 31, 512)
        # self.zero_pad1 = keras.layers.ZeroPadding2D()
        self.zero_pad1 = nn.ZeroPad2d(1)
        # self.conv = keras.layers.Conv2D(512, (4, 4),
        #                                 strides=1,
        #                                 kernel_initializer=initializer,
        #                                 use_bias=False)
        self.conv = nn.Conv2d(256, 512, (3, 3), stride=1, padding=1, bias=False)
        # self.batchnorm1 = keras.layers.BatchNormalization()
        self.batchnorm1 = nn.BatchNorm2d(512)
        self.leaky_relu = nn.LeakyReLU()

        # shape change from (batch_size, 31, 31, 512) to (batch_size, 30, 30, 1)
        # self.zero_pad2 = keras.layers.ZeroPadding2D()
        self.zero_pad2 = nn.ZeroPad2d(1)
        # self.last = keras.layers.Conv2D(1, (4, 4),
        #                                 strides=1,
        #                                 kernel_initializer=initializer)
        self.last = nn.Conv2d(512, 1,  (3, 3), stride=1, padding=1, bias=False)

    def forward(self, inputs):
        inp, target = inputs

        # concatenating the input and the target
        x = torch.concat([inp, target], dim=1)  # (bs, 64, 64, channels*2)

        x = self.down1(x)  # (bs, 32, 32, 64)
        x = self.down2(x)  # (bs, 16, 16, 128)
        x = self.down3(x)  # (bs, 8, 8, 256)

        x = self.zero_pad1(x)  # (bs, 10, 10, 256)
        x = self.conv(x)  # (bs, 7, 7, 512)
        x = self.batchnorm1(x)
        # x = tf.nn.leaky_relu(x)
        x = self.leaky_relu(x)

        x = self.zero_pad2(x)  # (bs, 9, 9, 512)
        # don't add a sigmoid activation here since
        # the loss function expects raw logits.
        x = self.last(x)  # (bs, 8, 8, 1)

        # x = tf.nn.sigmoid(x)

        return x

# class Discriminator(keras.Model):
#
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         initializer = tf.random_normal_initializer(0., 0.02)
#
#
#         # self.down1 = DiscDownsample(64, 4, False)
#         # self.down2 = DiscDownsample(128, 4)
#         # self.down3 = DiscDownsample(256, 4)
#
#         self.down1 = DiscDownsample(64, 4, False)
#         self.down2 = DiscDownsample(128, 4)
#         self.down3 = DiscDownsample(256, 4)
#
#         # we are zero padding here with 1 because we need our shape to
#         # go from (batch_size, 32, 32, 256) to (batch_size, 31, 31, 512)
#         self.zero_pad1 = keras.layers.ZeroPadding2D()
#         self.conv = keras.layers.Conv2D(512, (4, 4),
#                                         strides=1,
#                                         kernel_initializer=initializer,
#                                         use_bias=False)
#         self.batchnorm1 = keras.layers.BatchNormalization()
#
#         # shape change from (batch_size, 31, 31, 512) to (batch_size, 30, 30, 1)
#         self.zero_pad2 = keras.layers.ZeroPadding2D()
#         self.last = keras.layers.Conv2D(1, (4, 4),
#                                         strides=1,
#                                         kernel_initializer=initializer)
#
#     def call(self, inputs, training=None):
#         inp, target = inputs
#
#         # concatenating the input and the target
#         x = tf.concat([inp, target], axis=-1)  # (bs, 64, 64, channels*2)
#
#         x = self.down1(x, training=training)  # (bs, 32, 32, 64)
#         x = self.down2(x, training=training)  # (bs, 16, 16, 128)
#         x = self.down3(x, training=training)  # (bs, 8, 8, 256)
#
#         x = self.zero_pad1(x)  # (bs, 10, 10, 256)
#         x = self.conv(x)  # (bs, 7, 7, 512)
#         x = self.batchnorm1(x, training=training)
#         x = tf.nn.leaky_relu(x)
#
#         x = self.zero_pad2(x)  # (bs, 9, 9, 512)
#         # don't add a sigmoid activation here since
#         # the loss function expects raw logits.
#         x = self.last(x)  # (bs, 8, 8, 1)
#
#         # x = tf.nn.sigmoid(x)
#
#         return x
