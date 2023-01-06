import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# built in function parameters for load_data() that takes in a tupel (data set of two), the 
# training img and labels, and test ones.  i'd imagine it's a preset that splits them.

# image pixels are on 255 value channels (either rgb on each layer, or grayscale).
# we want that number to be between 1 and 0, so divide by 255
train_images, test_images = train_images / 255.0, test_images / 255.0

# assign string values to the label number value from data
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# looking at a single image in dataset
IMG_INDEX = 7 
plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)

# DATA SET NO LONGER WORKING FROM SOURCE, TIME TO CHECK IF THE NOTEBOOK VERSION OF THIS CODE WORKS