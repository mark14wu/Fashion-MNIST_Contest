import sys
from keras.models import Model
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.models import model_from_json

# commandline arguments
weights_filename = sys.argv[1]
image_filename = sys.argv[2]
json_filename = 'mobilenet-saved-models/architecture'

# load a model
json_string = open(json_filename).read()
model = model_from_json(json_string)

# load model weights
model.load_weights(weights_filename)

model.compile()

# load pictures
img = image.load_img(image_filename, target_size=(128, 128))
x = image.img_to_array(img)

# predict models
print(model.predict(x))