import numpy as np
import mnist_reader
from tqdm import tqdm
from scipy import misc
import tensorflow as tf


np.random.seed(2017)
tf.set_random_seed(2017)

X_train, y_train = mnist_reader.load_mnist('./data', kind='train')
X_test, y_test = mnist_reader.load_mnist('./data', kind='t10k')

height,width = 128,128


from keras.applications.mobilenet import MobileNet
from keras.layers import Input,Dense,Dropout,Lambda
from keras.models import Model
from keras import backend as K
import keras
import keras.optimizers
import keras.callbacks


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

class genCB(keras.callbacks.Callback):
    def __init__(self, val):
        self.val = val

    def on_epoch_end(self, epoch, logs={}):
        mtloss = np.mean([self.model.test_on_batch(data) for data in sample_data(self.val, once=True)])
        print('                                   val loss %f' % (mtloss,))

input_image = Input(shape=(height,width))
input_image_ = Lambda(lambda x: K.repeat_elements(K.expand_dims(x,3),3,3))(input_image)
base_model = MobileNet(input_tensor=input_image_, include_top=False, pooling='avg',input_shape=(128,128,3))
output = Dropout(0.5)(base_model.output)
predict = Dense(10, activation='softmax')(output)

model = Model(inputs=input_image, outputs=predict)
my_adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=my_adam, loss='sparse_categorical_crossentropy', metrics=['accuracy',f1])
model.summary()
tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./mobilenet_logs',\
 histogram_freq=0, batch_size=100, write_graph=True,\
  write_grads=False, write_images=True, embeddings_freq=0,\
   embeddings_layer_names=None, embeddings_metadata=None)


X_train = X_train.reshape((-1,28,28))
X_train = np.array([misc.imresize(x, (height,width)).astype(float) for x in tqdm(iter(X_train))])/255.

X_test = X_test.reshape((-1,28,28))
X_test = np.array([misc.imresize(x, (height,width)).astype(float) for x in tqdm(iter(X_test))])/255.

def random_reverse(x):
	if np.random.random() > 0.5:
		return x[:,::-1]
	else:
		return x

def data_generator(X,Y,batch_size=100):
	while True:
		idxs = np.random.permutation(len(X))
		X = X[idxs]
		Y = Y[idxs]
		p,q = [],[]
		for i in range(len(X)):
			p.append(random_reverse(X[i]))
			q.append(Y[i])
			if len(p) == batch_size:
				yield np.array(p),np.array(q)
				p,q = [],[]
		if p:
			yield np.array(p),np.array(q)
			p,q = [],[]



# model.fit_generator(data_generator(X_train,y_train), steps_per_epoch=600,\
#  epochs=500, validation_data=data_generator(X_test,y_test), validation_steps=100)

model.fit(X_train, y_train, batch_size=128, epochs=100, \
callbacks=[tensorboard_callback], shuffle=True, validation_data=(X_test, y_test))

