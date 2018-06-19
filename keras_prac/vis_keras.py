import keras
from keras.utils import plot_model

from keras.applications.resnet50 import ResNet50
model = ResNet50()
print(model.summary())
plot_model(model, to_file='model.png')
