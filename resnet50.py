'''
import keras
import keras.datasets

# importing fashion-MNIST dataset
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# building up ResNet50 model
import keras_resnet.models
shape, classes = (28, 28, 1), 10
x = keras.layers.Input(shape)
model = keras_resnet.models.ResNet50(x, classes=classes)
model.compile("adam", "categorical_crossentropy", ["accuracy"])
y_train = keras.utils.np_utils.to_categorical(y_train)

# fitting the model
model.fit(x_train, y_train, epochs=10, batch_size=256)
score = model.evaluate(x_test, y_test, batch_size=256)
print(score)
'''

from numpy import random
random.seed(42)  # @UndefinedVariable

from keras.datasets import fashion_mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from resnet import Residual

batch_size = 128
nb_classes = 10
nb_epoch = 1

img_rows, img_cols = 28, 28
pool_size = (2, 2)
kernel_size = (3, 3)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_dim_ordering() == 'th':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
X_test = X_test.astype('float32')
x_train /= 255
X_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Model
input_var = Input(shape=input_shape)

conv1 = Convolution2D(64, kernel_size[0], kernel_size[1],
                      border_mode='same', activation='relu')(input_var)
# conv1 = MaxPooling2D(pool_size=pool_size)(conv1)
conv2 = Convolution2D(8, kernel_size[0], kernel_size[1],
                      border_mode='same', activation='relu')(conv1)

resnet = conv2
for _ in range(5):
    resnet = Residual(Convolution2D(8, kernel_size[0], kernel_size[1],
                                  border_mode='same'))(resnet)
    resnet = Activation('relu')(resnet)

mxpool = MaxPooling2D(pool_size=pool_size)(resnet)
flat = Flatten()(mxpool)
dropout = Dropout(0.5)(flat)
softmax = Dense(nb_classes, activation='softmax')(dropout)

model = Model(input=[input_var], output=[softmax])
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
model.save('fashion_mnist_model.h5')
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])