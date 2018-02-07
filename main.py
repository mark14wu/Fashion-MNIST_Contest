import keras
import keras.datasets
from vgg import vgg_fm

# importing fashion-MNIST dataset
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
y_train = keras.utils.np_utils.to_categorical(y_train)
y_test = keras.utils.np_utils.to_categorical(y_test)

# building up model
input_shape = x_train.shape[1:]
print(input_shape)
model = vgg_fm((28, 28, 1))

# compiling and fitting the model
epochs = 100
batch_size = 256

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
model.fit(x=x_train, y=y_train, \
validation_data=(x_test, y_test), \
batch_size=batch_size, epochs=epochs)
model.fit(x_train, y_train, epochs=10, batch_size=256)
score = model.evaluate(x_test, y_test, batch_size=256)
print(score)