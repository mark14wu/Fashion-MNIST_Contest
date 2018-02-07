import keras
import keras.datasets
from vgg import vgg_fm
from keras.callbacks import ModelCheckpoint
from callback import TargetStopping

# importing fashion-MNIST dataset
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
y_train = keras.utils.np_utils.to_categorical(y_train)
y_test = keras.utils.np_utils.to_categorical(y_test)

# building up model
input_shape = x_train.shape[1:]
print(input_shape)
# model = vgg_fm((28, 28, 1))
model = vgg_fm(input_shape)
model.summary()
model_name = 'vgg'

# compiling and fitting the model

# setting up parameters
epochs = 100
batch_size = 256

# # starting GPU manager
# from manager import GPUManager
# gm=GPUManager()


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])

# with gm.auto_choice():
model.fit(x=x_train, y=y_train, \
validation_data=(x_test, y_test), \
batch_size=batch_size, epochs=epochs, \
callbacks=[TargetStopping(filepath=model_name+'.h5',\
monitor='val_acc',mode='max',target=0.94),\
ModelCheckpoint(filepath=model_name+'.h5',save_best_only=True,\
monitor='val_acc')])

score = model.evaluate(x_test, y_test, batch_size=256)
print(score)