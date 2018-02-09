import sys
from keras.models import Model
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# commandline arguments
model_filename = sys.argv[1]
image_filename = sys.argv[2]

# load a model
model = load_model(model_filename)

# load pictures
img = image.load_img(image_filename, target_size=(128, 128))
x = image.img_to_array(img)

# predict models
print(model.predict(x))