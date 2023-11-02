# %% [markdown]
# # IAM Writer Recognition

# %% [markdown]
# This notebook is pretty much a translation of the "[handwriting_recognition](https://github.com/priya-dwivedi/Deep-Learning/tree/master/handwriting_recognition)" notebook by [Priyanka Dwivedi](https://github.com/priya-dwivedi). I have chosen to rewrite it differently here as to make it easier to follow, for my own better understanding, and for others who wish to learn from it.

# %% [markdown]
# The goal of the notebook is to use the method explained in the paper [DeepWriter: A Multi-Stream Deep CNN for Text-independent Writer Identification](https://arxiv.org/abs/1606.06472) to identify the writer (author) of a text based on their writing styles. To do so, we'll use the [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database/download-the-iam-handwriting-database). Please make sure the dataset has been correctly set up before executing the notebook as outlined [here](https://github.com/diegocasmo/iam_writer_recognition/tree/master/data).

# %% [markdown]
# # Reading The Dataset

# %% [markdown]
# The first step is to create a dictionary which will map each form ID (sentence) to a writer. This information is available in the ``forms.txt`` file, where each line (except for the first 16 lines, which are documentation) defines the form ID at index ``0``, and its writer at index ``1``.

# %%
# Create a dictionary to store each form ID and its writer
import os
from itertools import islice

form_writer = {}
forms_file_path = "../data/forms.txt"
with open(forms_file_path) as f:
    for line in islice(f, 16, None):
        line_list = line.split(' ')
        form_id = line_list[0]
        writer = line_list[1]
        form_writer[form_id] = writer

# %% [markdown]
# Visualize dictionary (as array for simplicity):

# %%
list(form_writer.items())[0:5]

# %% [markdown]
# For efficiency reasons,  we'll select the 50 most common writers from the dictionary we have created, and the rest of the notebook will only focus on them (as opposed to using the 221 authors present in the dataset).

# %%
# Select the 50 most common writer

from collections import Counter

top_writers = []
num_writers = 50
writers_counter = Counter(form_writer.values())
for writer_id,_ in writers_counter.most_common(num_writers):
    top_writers.append(writer_id)

# %% [markdown]
# Visualize the writer id of the top 50 writers:

# %%
print(top_writers[0:5])

# %% [markdown]
# From the 50 most common writers we have selected, we'll now need to select the forms (sentences) they have written:

# %%
top_forms = []
for form_id, author_id in form_writer.items():
    if author_id in top_writers:
        top_forms.append(form_id)

# %% [markdown]
# Visualize the form id of the top 50 writers:

# %%
print(top_forms[0:5])

# %% [markdown]
# Create a temp directory which contains only the sentences of the forms selected above:

# %%
import os
import glob
import shutil

# Create temp directory to save writers' forms in (assumes files have already been copied if the directory exists)
temp_sentences_path = "../data/temp_sentences"
if not os.path.exists(temp_sentences_path):
    os.makedirs(temp_sentences_path)
    # Copy forms that belong to the top 50 most common writers to the temp directory
    original_sentences_path = "../data/sentences/**/**/*.png"
    for file_path in glob.glob(original_sentences_path):
        image_name = file_path.split('/')[-1]  
        file_name, _ = os.path.splitext(image_name)
        form_id = '-'.join(file_name.split('-')[0:2])
        if form_id in top_forms:
            shutil.copy2(file_path, temp_sentences_path + "/" + image_name)

# %% [markdown]
# Create arrays of file inputs (a form) and their respective targets (a writer id):

# %%
import numpy as np

img_files = np.zeros((0), dtype=np.str)
img_targets = np.zeros((0), dtype=np.str)
path_to_files = os.path.join(temp_sentences_path, '*')
for file_path in glob.glob(path_to_files):
    img_files = np.append(img_files, file_path)
    file_name, _ = os.path.splitext(file_path.split('/')[-1]  )
    form_id = '-'.join(file_name.split('-')[0:2])
    for key in form_writer:
        if key == form_id:
            img_targets = np.append(img_targets, form_writer[form_id])

# %% [markdown]
# Visualize the form -> writer id arrays:

# %%
print(img_files[0:5])
print(img_targets[0:5])

# %% [markdown]
# Visualize dataset's images:

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

for file_name in img_files[:2]:
    img = mpimg.imread(file_name)
    plt.figure(figsize = (10,10))
    plt.imshow(img, cmap ='gray')

# %% [markdown]
# Encode writers with a value between 0 and ``n_classes-1``:

# %%
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(img_targets)
encoded_img_targets = encoder.transform(img_targets)

print("Writer ID        : ", img_targets[:2])
print("Encoded writer ID: ", encoded_img_targets[:2])

# %% [markdown]
# Split dataset into train, validation, and tests sets:

# %%
from sklearn.model_selection import train_test_split

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(img_files, encoded_img_targets, test_size=0.2, shuffle = True)

# Further split training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle = True)

print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)

# %% [markdown]
# Define a couple of constants that will be used throughout the model:

# %%
CROP_SIZE = 113
NUM_LABELS = 50
BATCH_SIZE = 16

# %% [markdown]
# As suggested in the paper, the input to the model are not unique sentences but rather random patches cropped from each sentence. The ``get_augmented_sample`` method is in charge of doing so by resizing each sentence's height to ``113`` pixels, and its width such that original aspect ratio is maintained. Finally, from the resized image, patches of ``113x113`` are randomly cropped.

# %%
from sklearn.utils import shuffle
from PIL import Image
import random

def get_augmented_sample(sample, label, sample_ratio):
    # Get current image details
    img = Image.open(sample)
    img_width = img.size[0]
    img_height = img.size[1]

    # Compute resize dimensions such that aspect ratio is maintained
    height_fac = CROP_SIZE / img_height
    size = (int(img_width * height_fac), CROP_SIZE)

    # Resize image 
    new_img = img.resize((size), Image.ANTIALIAS)
    new_img_width = new_img.size[0]
    new_img_height = new_img.size[1]

    # Generate a random number of crops of size 113x113 from the resized image
    x_coord = list(range(0, new_img_width - CROP_SIZE))
    num_crops = int(len(x_coord) * sample_ratio)
    random_x_coord = random.sample(x_coord, num_crops)
    
    # Create augmented images (cropped forms) and map them to a label (writer)
    images = []
    labels = []
    for x in random_x_coord:
        img_crop = new_img.crop((x, 0, x + CROP_SIZE, CROP_SIZE))
        # Transform image to an array of numbers
        images.append(np.asarray(img_crop))
        labels.append(label)

    return (images, labels)

# %% [markdown]
# Let's visualize what the ``get_augmented_sample`` method does by augmenting one sample from the training set. Let's first take a look at how the original image looks like:

# %%
sample, label = X_train[0], y_train[0]
img = mpimg.imread(sample)
plt.figure(figsize = (10,10))
plt.imshow(img, cmap ='gray')
print("Label: ", label)

# %% [markdown]
# A now, let's augment it and see the result:

# %%
images, labels = get_augmented_sample(sample, label, 0.1)

# %% [markdown]
# The ``labels`` returned by the ``get_augmented_sample`` is simply the label of the original image for each cropped patch:

# %%
print(labels)
print("Num of labels: ", len(labels))

# %% [markdown]
# And the ``images`` returned by it are the random patches created from the original image (only two samples shown for simplicity):

# %%
print(len(images))
plt.imshow(images[0], cmap ='gray')

# %%
plt.imshow(images[1], cmap ='gray')

# %% [markdown]
# The model uses a generator in order to be able to call ``get_augmented_sample`` when training the model:

# %%
import operator
from functools import reduce
from keras.utils import to_categorical

def generate_data(samples, labels, batch_size, sample_ratio):
    while 1: 
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:(offset + batch_size)]
            batch_labels = labels[offset:(offset + batch_size)]
            
            # Augment each sample in batch
            augmented_batch_samples = []
            augmented_batch_labels = []
            for i in range(len(batch_samples)):
                sample = batch_samples[i]
                label = batch_labels[i]
                augmented_samples, augmented_labels = get_augmented_sample(sample, label, sample_ratio)
                augmented_batch_samples.append(augmented_samples)
                augmented_batch_labels.append(augmented_labels)

            # Flatten out samples and labels
            augmented_batch_samples = reduce(operator.add, augmented_batch_samples)
            augmented_batch_labels = reduce(operator.add, augmented_batch_labels)
            
            # Reshape input format
            X_train = np.array(augmented_batch_samples)
            X_train = X_train.reshape(X_train.shape[0], CROP_SIZE, CROP_SIZE, 1)

            # Transform input to float and normalize
            X_train = X_train.astype('float32')
            X_train /= 255

            # Encode y
            y_train = np.array(augmented_batch_labels)
            y_train = to_categorical(y_train, NUM_LABELS)

            yield X_train, y_train

# %% [markdown]
# Create training, validation, and test generators:

# %%
train_generator = generate_data(X_train, y_train, BATCH_SIZE, 0.3)
validation_generator = generate_data(X_val, y_val, BATCH_SIZE, 0.3)
test_generator = generate_data(X_test, y_test, BATCH_SIZE, 0.1)

# %%
import tensorflow as tf

config = tf.ConfigProto()
tf.Session(config = config)

# %%
def resize_image(img):
    size = round(CROP_SIZE/2)
    return tf.image.resize_images(img, [size, size])

# %% [markdown]
# The model used is exactly the same as the one in the "[handwriting_recognition](https://github.com/priya-dwivedi/Deep-Learning/tree/master/handwriting_recognition)" notebook by [Priyanka Dwivedi](https://github.com/priya-dwivedi):

# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.optimizers import Adam
from keras import metrics

model = Sequential()

# Define network input shape
model.add(ZeroPadding2D((1, 1), input_shape=(CROP_SIZE, CROP_SIZE, 1)))
# Resize images to allow for easy computation
model.add(Lambda(resize_image)) 

# CNN model - Building the model suggested in paper
model.add(Convolution2D(filters= 32, kernel_size =(5,5), strides= (2, 2), padding='same', name='conv1'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1'))

model.add(Convolution2D(filters= 64, kernel_size =(3, 3), strides= (1, 1), padding='same', name='conv2'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2'))

model.add(Convolution2D(filters= 128, kernel_size =(3, 3), strides= (1, 1), padding='same', name='conv3'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3'))


model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(512, name='dense1'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256, name='dense2'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(NUM_LABELS, name='output'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])

print(model.summary())

# %% [markdown]
# Next, the model is trained for 20 epochs and the models obtained after each epoch are saved to the ``./model_checkpoints`` directory

# %%
from keras.callbacks import ModelCheckpoint

# Create directory to save checkpoints at
model_checkpoints_path = "./model_checkpoints"
if not os.path.exists(model_checkpoints_path):
    os.makedirs(model_checkpoints_path)
    
# Save model after every epoch using checkpoints
create_checkpoint = ModelCheckpoint(
    filepath = "./model_checkpoints/check_{epoch:02d}_{val_loss:.4f}.hdf5",
    verbose = 1,
    save_best_only = False
)

# Fit model using generators
history_object = model.fit_generator(
    train_generator, 
    steps_per_epoch = round(len(X_train) / BATCH_SIZE),
    validation_data = validation_generator,
    validation_steps = round(len(X_val) / BATCH_SIZE),
    epochs = 20,
    verbose = 1,
    callbacks = [create_checkpoint]
)

# %% [markdown]
# Load a saved model weights and use them to predict labels in the test set:

# %%
model_weights_path = "./model_checkpoints/model_weights.hdf5"
if model_weights_path:
    model.load_weights(model_weights_path)
    scores = model.evaluate_generator(test_generator, steps=round(len(X_test)/BATCH_SIZE))
    print("Accuracy: ", scores[1])
else:
    print("Set model weights file to load in the 'model_weights_path' variable")


