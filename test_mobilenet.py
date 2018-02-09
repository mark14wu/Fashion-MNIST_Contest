import sys
from keras.models import Model

weights_filename = sys.argv[1]
image_filename = sys.argv[2]


# construct and load a model
model = Model(inputs=input_image, outputs=predict)
model.load_weights(weights_filename)