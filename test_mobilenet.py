import sys
from keras.models import Model
import numpy as np
from keras.preprocessing import image

# commandline arguments
weights_filename = sys.argv[1]
image_filename = sys.argv[2]

# construct and load a model
model = Model(inputs=input_image, outputs=predict)
model.load_weights(weights_filename)
model.compile(optimizer=my_adam, loss='sparse_categorical_crossentropy', metrics=['accuracy',f1])

# load pictures
img = image.load_img(image_filename, target_size=(128, 128))
x = image.img_to_array(img)

# predict models
print(model.predict(x))