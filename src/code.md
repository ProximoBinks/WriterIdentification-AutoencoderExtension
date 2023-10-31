# IAM Writer Recognition

```python
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
```

Visualize dictionary (as array for simplicity):
```python
list(form_writer.items())[0:5]
print("Number of form-writer pairs:", len(form_writer))
print(list(form_writer.items())[0:5])
print("Sample form-writer mappings:", list(form_writer.items())[:5])
```

For efficiency reasons,  we'll select the 50 most common writers from the dictionary we have created, and the rest of the notebook will only focus on them (as opposed to using the 221 authors present in the dataset).

```python
# Select the 50 most common writer

from collections import Counter

top_writers = []
num_writers = 50
writers_counter = Counter(form_writer.values())
for writer_id,_ in writers_counter.most_common(num_writers):
    top_writers.append(writer_id)
```

Visualize the writer id of the top 50 writers:

```python
print("Top writer IDs:", top_writers[0:5])
print(top_writers[0:5])
```

From the 50 most common writers we have selected, we'll now need to select the forms (sentences) they have written:

```python
top_forms = []
for form_id, author_id in form_writer.items():
    if author_id in top_writers:
        top_forms.append(form_id)
```

Visualize the form id of the top 50 writers:

```python
print("Number of top forms:", len(top_forms))
print("Sample form IDs:", top_forms[:5])
print(top_forms[0:5])
```

Create a temp directory which contains only the sentences of the forms selected above:


```python
import os
import glob
import shutil

# Create temp directory to save writers' forms in (assumes files have already been copied if the directory exists)
temp_sentences_path = "../data/temp_sentences"
if not os.path.exists(temp_sentences_path):
    os.makedirs(temp_sentences_path)

# Debugging Line 4: Check if 'top_forms' is correctly set
print(f"Top Forms: {top_forms}")

original_sentences_path = os.path.join("..", "data", "sentences", "*", "*", "*.png")

# Debugging Line 5: Verify the Paths
print("Files found:", glob.glob(original_sentences_path)[:5])

for file_path in glob.glob(original_sentences_path):
    image_name = file_path.split(os.path.sep)[-1]  # Use os.path.sep for cross-platform compatibility
    form_id = image_name.split('-')[0] + '-' + image_name.split('-')[1]

    if form_id in top_forms:
        # Debugging Line 6: Check if Files are Copied
        print(f"Copying file {file_path} to {temp_sentences_path}/{image_name}")
        try:
            shutil.copy(file_path, os.path.join(temp_sentences_path, image_name))
        except Exception as e:
            print(f"Failed to copy {file_path}. Error: {e}")

```

Create arrays of file inputs (a form) and their respective targets (a writer id):

```python
import os
import glob
import shutil
import numpy as np

img_files = np.zeros((0), dtype=str)
img_targets = []

path_to_files = os.path.join(temp_sentences_path, '*')
for file_path in glob.glob(path_to_files):
    img_files = np.append(img_files, file_path)
    file_name, _ = os.path.splitext(file_path.split(os.path.sep)[-1])
    form_id = '-'.join(file_name.split('-')[0:2])
    if form_id in form_writer:
        img_targets.append(form_writer[form_id])

# Convert img_targets to a NumPy array
img_targets = np.array(img_targets)

# Debugging Line 7: Validate Array Populations
print("Array lengths:", len(img_files), len(img_targets))

```

    Array lengths: 4909 4909
    

Visualize the form -> writer id arrays:


```python
print(f"Checking path: {path_to_files}")
files_found = glob.glob(path_to_files)
print(f"Found {len(files_found)} files.")

print(img_files[0:5])
print(img_targets[0:5])
```

Visualize dataset's images:


```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

for file_name in img_files[:2]:
    img = mpimg.imread(file_name)
    plt.figure(figsize = (10,10))
    plt.imshow(img, cmap ='gray')
```

![png](solution_final_files/solution_final_23_0.png)
  
![png](solution_final_files/solution_final_23_1.png)

Encode writers with a value between 0 and ``n_classes-1``:

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoded_img_targets = encoder.fit_transform(img_targets)

print("Writer ID        : ", img_targets[:2])
print("Encoded writer ID: ", encoded_img_targets[:2])
```

    Writer ID        :  ['000' '000']
    Encoded writer ID:  [0 0]

Split dataset into train, validation, and tests sets:

```python
from sklearn.model_selection import train_test_split

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(img_files, encoded_img_targets, test_size=0.2, shuffle = True)

# Further split training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle = True)

print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)
```

    (3141,) (786,) (982,)
    (3141,) (786,) (982,)

Define a couple of constants that will be used throughout the model:

```python
CROP_SIZE = 113
NUM_LABELS = 50
BATCH_SIZE = 16
```

As suggested in the paper, the input to the model are not unique sentences but rather random patches cropped from each sentence. The ``get_augmented_sample`` method is in charge of doing so by resizing each sentence's height to ``113`` pixels, and its width such that original aspect ratio is maintained. Finally, from the resized image, patches of ``113x113`` are randomly cropped.

```python
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
```

Let's visualize what the ``get_augmented_sample`` method does by augmenting one sample from the training set. Let's first take a look at how the original image looks like:

```python
sample, label = X_train[0], y_train[0]
img = mpimg.imread(sample)
plt.figure(figsize = (10,10))
plt.imshow(img, cmap ='gray')
print("Label: ", label)
```

    Label:  21
 
![png](solution_final_files/solution_final_33_1.png)

A now, let's augment it and see the result:

```python
images, labels = get_augmented_sample(sample, label, 0.1)
```

The ``labels`` returned by the ``get_augmented_sample`` is simply the label of the original image for each cropped patch:

```python
print(labels)
print("Num of labels: ", len(labels))
```

    [21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21]
    Num of labels:  138

And the ``images`` returned by it are the random patches created from the original image (only two samples shown for simplicity):

```python
print(len(images))
plt.imshow(images[0], cmap ='gray')
```

```python
plt.imshow(images[1], cmap ='gray')
```

The model uses a generator in order to be able to call ``get_augmented_sample`` when training the model:

```python
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
```

Create training, validation, and test generators:

```python
train_generator = generate_data(X_train, y_train, BATCH_SIZE, 0.3)
validation_generator = generate_data(X_val, y_val, BATCH_SIZE, 0.3)
test_generator = generate_data(X_test, y_test, BATCH_SIZE, 0.1)
```

```python
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

```python
def resize_image(img):
    size = round(CROP_SIZE/2)
    return tf.image.resize(img, [size, size])
```

```python
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
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     zero_padding2d (ZeroPadding  (None, 115, 115, 1)      0         
     2D)                                                             
                                                                     
     lambda (Lambda)             (None, 56, 56, 1)         0         
                                                                     
     conv1 (Conv2D)              (None, 28, 28, 32)        832       
                                                                     
     activation (Activation)     (None, 28, 28, 32)        0         
                                                                     
     pool1 (MaxPooling2D)        (None, 14, 14, 32)        0         
                                                                     
     conv2 (Conv2D)              (None, 14, 14, 64)        18496     
                                                                     
     activation_1 (Activation)   (None, 14, 14, 64)        0         
                                                                     
     pool2 (MaxPooling2D)        (None, 7, 7, 64)          0         
                                                                     
     conv3 (Conv2D)              (None, 7, 7, 128)         73856     
                                                                     
     activation_2 (Activation)   (None, 7, 7, 128)         0         
                                                                     
     pool3 (MaxPooling2D)        (None, 3, 3, 128)         0         
                                                                     
     flatten (Flatten)           (None, 1152)              0         
                                                                     
     dropout (Dropout)           (None, 1152)              0         
                                                                     
     dense1 (Dense)              (None, 512)               590336    
                                                                     
     activation_3 (Activation)   (None, 512)               0         
                                                                     
     dropout_1 (Dropout)         (None, 512)               0         
                                                                     
     dense2 (Dense)              (None, 256)               131328    
                                                                     
     activation_4 (Activation)   (None, 256)               0         
                                                                     
     dropout_2 (Dropout)         (None, 256)               0         
                                                                     
     output (Dense)              (None, 50)                12850     
                                                                     
     activation_5 (Activation)   (None, 50)                0         
                                                                     
    =================================================================
    Total params: 827,698
    Trainable params: 827,698
    Non-trainable params: 0
    _________________________________________________________________
    None
    

Next, the model is trained for 20 epochs and the models obtained after each epoch are saved to the ``./model_checkpoints`` directory

```python
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
```

    C:\Users\User\AppData\Local\Temp\ipykernel_46608\3366608101.py:16: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.
      history_object = model.fit_generator(
    C:\Users\User\AppData\Local\Temp\ipykernel_46608\3032259505.py:16: DeprecationWarning: ANTIALIAS is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.LANCZOS instead.
      new_img = img.resize((size), Image.ANTIALIAS)

    Epoch 1/20
    196/196 [==============================] - ETA: 0s - loss: 3.4459 - acc: 0.2137
    Epoch 1: saving model to ./model_checkpoints\check_01_2.9294.hdf5
    196/196 [==============================] - 334s 2s/step - loss: 3.4459 - acc: 0.2137 - val_loss: 2.9294 - val_acc: 0.2694
    Epoch 2/20
    196/196 [==============================] - ETA: 0s - loss: 2.7850 - acc: 0.2656
    Epoch 2: saving model to ./model_checkpoints\check_02_2.3100.hdf5
    196/196 [==============================] - 388s 2s/step - loss: 2.7850 - acc: 0.2656 - val_loss: 2.3100 - val_acc: 0.3527
    Epoch 20/20
    196/196 [==============================] - ETA: 0s - loss: 0.8745 - acc: 0.7315
    Epoch 20: saving model to ./model_checkpoints\check_20_0.7311.hdf5
    196/196 [==============================] - 359s 2s/step - loss: 0.8745 - acc: 0.7315 - val_loss: 0.7311 - val_acc: 0.7771
    
Load a saved model weights and use them to predict labels in the test set:

```python
model_weights_path = "./model_checkpoints/model_weights.hdf5"
if model_weights_path:
    model.load_weights(model_weights_path)
    scores = model.evaluate_generator(test_generator, steps=round(len(X_test)/BATCH_SIZE))
    print("Accuracy: ", scores[1])
else:
    print("Set model weights file to load in the 'model_weights_path' variable")
```

    Accuracy:  0.704222559928894