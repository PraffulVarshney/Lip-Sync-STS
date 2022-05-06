from keras.models import load_model
import numpy as np
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Conv2D, Conv3D, BatchNormalization, Activation, \
						Concatenate, AvgPool2D, Input, MaxPool2D, UpSampling2D, Add, \
						ZeroPadding2D, ZeroPadding3D, Lambda, Reshape, Flatten, LeakyReLU
from keras_contrib.layers import InstanceNormalization
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
import cv2
import os
import librosa
import scipy
from keras.utils import plot_model
import tensorflow as tf
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras import backend as K

def contrastive_loss(y_true, y_pred):
	margin = 1.
	loss = (1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(0., margin - y_pred))  #1-y_true = yi, y_pred = d
	return K.mean(loss)

def conv_block(x, num_filters, kernel_size=3, strides=2, padding='same'):
	x = Conv2D(filters=num_filters, kernel_size= kernel_size, 
					strides=strides, padding=padding)(x)
	x = InstanceNormalization()(x)  # to prevent instance-specific mean and covariance shift simplifying the learning process
	x = LeakyReLU(alpha=.2)(x)
	return x

def create_model(args, mel_step_size):
	############# encoder for face/identity
	input_face = Input(shape=(args.img_size, args.img_size, 3), name="input_face_disc")

	x = conv_block(input_face, 64, 7)
	x = conv_block(x, 128, 5)
	x = conv_block(x, 256, 3)
	x = conv_block(x, 512, 3)
	x = conv_block(x, 512, 3)
	x = Conv2D(filters=512, kernel_size=3, strides=1, padding="valid")(x)
	face_embedding = Flatten() (x)

	############# encoder for audio
	input_audio = Input(shape=(80, mel_step_size, 1), name="input_audio")

	x = conv_block(input_audio, 32, strides=1)
	x = conv_block(x, 64, strides=3)	#27X9
	x = conv_block(x, 128, strides=(3, 1)) 		#9X9
	x = conv_block(x, 256, strides=3)	#3X3
	x = conv_block(x, 512, strides=1, padding='valid')	#1X1
	x = conv_block(x, 512, 1, strides=1)

	audio_embedding = Flatten() (x)

	# L2-normalize before taking L2 distance
	l2_normalize = Lambda(lambda x: K.l2_normalize(x, axis=1)) 
	face_embedding = l2_normalize(face_embedding)
	audio_embedding = l2_normalize(audio_embedding)

	d = Lambda(lambda x: K.sqrt(K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True))) ([face_embedding,
																		audio_embedding])

	model = Model(inputs=[input_face, input_audio], outputs=[d])

	model.summary()



