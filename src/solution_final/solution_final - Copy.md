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

```python
list(form_writer.items())[0:5]
print("Number of form-writer pairs:", len(form_writer))
print(list(form_writer.items())[0:5])
print("Sample form-writer mappings:", list(form_writer.items())[:5])
```

```python
# Select the 50 most common writer

from collections import Counter

top_writers = []
num_writers = 50
writers_counter = Counter(form_writer.values())
for writer_id,_ in writers_counter.most_common(num_writers):
    top_writers.append(writer_id)
```

```python
print("Top writer IDs:", top_writers[0:5])
print(top_writers[0:5])
```

```python
top_forms = []
for form_id, author_id in form_writer.items():
    if author_id in top_writers:
        top_forms.append(form_id)
```

```python
print("Number of top forms:", len(top_forms))
print("Sample form IDs:", top_forms[:5])
print(top_forms[0:5])
```

```python
import os
import glob
import shutil

# Create temp directory to save writers' forms in (assumes files have already been copied if the directory exists)
temp_sentences_path = "../data/temp_sentences"
if not os.path.exists(temp_sentences_path):
    os.makedirs(temp_sentences_path)

# Debugging Line 4: Check if 'top_forms' is correctly set
#print(f"Top Forms: {top_forms}")

original_sentences_path = os.path.join("..", "data", "sentences", "*", "*", "*.png")

# Debugging Line 5: Verify the Paths
#print("Files found:", glob.glob(original_sentences_path)[:5])

for file_path in glob.glob(original_sentences_path):
    image_name = file_path.split(os.path.sep)[-1]  # Use os.path.sep for cross-platform compatibility
    form_id = image_name.split('-')[0] + '-' + image_name.split('-')[1]

    if form_id in top_forms:
        # Debugging Line 6: Check if Files are Copied
        #print(f"Copying file {file_path} to {temp_sentences_path}/{image_name}")
        try:
            shutil.copy(file_path, os.path.join(temp_sentences_path, image_name))
        except Exception as e:
            print(f"Failed to copy {file_path}. Error: {e}")

```

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

```python
print(f"Checking path: {path_to_files}")
files_found = glob.glob(path_to_files)
print(f"Found {len(files_found)} files.")

print(img_files[0:5])
print(img_targets[0:5])
```

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

for file_name in img_files[:2]:
    img = mpimg.imread(file_name)
    plt.figure(figsize = (10,10))
    plt.imshow(img, cmap ='gray')
```

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoded_img_targets = encoder.fit_transform(img_targets)

print("Writer ID        : ", img_targets[:2])
print("Encoded writer ID: ", encoded_img_targets[:2])
```

```python
from sklearn.model_selection import train_test_split

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(img_files, encoded_img_targets, test_size=0.2, shuffle = True)

# Further split training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle = True)

print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)
```

```python
CROP_SIZE = 113
NUM_LABELS = 50
BATCH_SIZE = 16
```

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
    new_img = img.resize(size, Image.LANCZOS)
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

```python
sample, label = X_train[0], y_train[0]
img = mpimg.imread(sample)
plt.figure(figsize = (10,10))
plt.imshow(img, cmap ='gray')
print("Label: ", label)
```

```python
images, labels = get_augmented_sample(sample, label, 0.1)
```

```python
print(labels)
print("Num of labels: ", len(labels))
```

```python
print(len(images))
plt.imshow(images[0], cmap ='gray')
```

```python
plt.imshow(images[1], cmap ='gray')
```

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

```python
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# Create directory to save checkpoints at
model_checkpoints_path = "./model_checkpoints"
if not os.path.exists(model_checkpoints_path):
    os.makedirs(model_checkpoints_path)

# Save model after every epoch using checkpoints
create_checkpoint = ModelCheckpoint(
    filepath="./model_checkpoints/check_{epoch:02d}_{val_loss:.4f}.hdf5",
    verbose=1,
    save_best_only=False
)

# Early stopping to stop the training if the validation loss does not improve for 5 epochs
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

# Fit model using generators
history_object = model.fit(
    train_generator,
    steps_per_epoch=round(len(X_train) / BATCH_SIZE),
    validation_data=validation_generator,
    validation_steps=round(len(X_val) / BATCH_SIZE),
    epochs=20,
    verbose=1,
    callbacks=[create_checkpoint, early_stopping]
)
```

```python
# Save only the model's weights
model.save_weights('my_model_weights.h5')
```

```python
model_weights_path = "./my_model_weights.h5"
if model_weights_path:
    model.load_weights(model_weights_path)
    scores = model.evaluate(test_generator, steps=round(len(X_test)/BATCH_SIZE))
    print("Accuracy: ", scores[1])
else:
    print("Set model weights file to load in the 'model_weights_path' variable")
```

```python
model.save('my_model.h5')
```

```python
from keras.models import load_model

# Path to the saved entire model
model_path = "./my_model.h5"

if model_path:
    # Load the entire saved model
    model = load_model(model_path)
    
    # Evaluate the model using the test generator
    scores = model.evaluate(test_generator, steps=round(len(X_test)/BATCH_SIZE))
    
    print("Accuracy: ", scores[1])
else:
    print("Set model file to load in the 'model_path' variable")

```

```python
def autoencoder_data_generator(samples, batch_size):
    while 1: 
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset + batch_size]
            
            images = []
            for sample in batch_samples:
                img = Image.open(sample)
                img = img.resize((113, 113))  # Resize to your input size
                img_array = np.asarray(img)
                images.append(img_array)
            
            X_train = np.array(images)
            X_train = X_train.reshape(X_train.shape[0], 113, 113, 1)
            X_train = X_train.astype('float32')
            X_train /= 255
            
            yield X_train, X_train  # x and y are the same for an autoencoder

# Create autoencoder data generators
autoencoder_train_generator = autoencoder_data_generator(X_train, 16)
autoencoder_val_generator = autoencoder_data_generator(X_val, 16)

# Now you can use these in your autoencoder training:
autoencoder.fit(autoencoder_train_generator, epochs=50, 
                steps_per_epoch=len(X_train) // 16,
                validation_data=autoencoder_val_generator,
                validation_steps=len(X_val) // 16)

```

```python
from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model

# Assuming each image has a shape of (113, 113, 1)
input_shape = (113, 113, 1)
encoding_dim = 64  # size of the encoded representations

# Encoder
input_img = Input(shape=input_shape)
flattened = Flatten()(input_img)
encoded = Dense(encoding_dim, activation='relu')(flattened)

# Decoder
decoded = Dense(np.prod(input_shape), activation='sigmoid')(encoded)
decoded = Reshape(input_shape)(decoded)

# Full autoencoder model
autoencoder = Model(input_img, decoded)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# You may need to adjust your data generators to work with the autoencoder.
# For now, I'm assuming you have suitable `autoencoder_train_generator` and `autoencoder_val_generator`.
autoencoder.fit(autoencoder_train_generator, epochs=50, 
                steps_per_epoch=len(X_train) // 16,
                validation_data=autoencoder_val_generator,
                validation_steps=len(X_val) // 16)
```

```python
# Encoder model for feature extraction
encoder = Model(input_img, encoded)
```

```python
# Assume you have some 'normal_data_points' and one 'new_data_point'
normal_data_points = np.random.rand(100, 113, 113, 1)  # I need to replace with my actual normal data
new_data_point = np.random.rand(1, 113, 113, 1)  # I need to replace with my actual new data point

# Encode the normal data points to get their latent space representations
encoded_normal_data_points = encoder.predict(normal_data_points)

# Encode the new data point to get its latent space representation
encoded_new_data_point = encoder.predict(new_data_point)

# Calculate the centroid of the normal data points in the latent space
centroid = np.mean(encoded_normal_data_points, axis=0)

# Calculate the distance between the new data point and the centroid
distance = np.linalg.norm(encoded_new_data_point - centroid)

# Define a threshold for anomaly detection (this should be based on your specific case)
threshold = 10.0

# Detect if the new data point is an anomaly
if distance > threshold:
    print("Anomaly detected")
else:
    print("No anomaly detected")
```

```python
# Assuming original_model and test_data_point are defined and preprocessed
# Define your confidence_threshold
original_model = model
confidence_threshold = 0.8

# Make a prediction with the original classifier
original_prediction = original_model.predict(test_data_point)

# If prediction confidence is low, check for the new class
if max(original_prediction[0]) < confidence_threshold:
    # Encode the test data point to get its latent space representation
    encoded_test_data_point = encoder.predict(test_data_point)
    
    # Calculate the 'difference' based on your specific logic
    # For instance, using Euclidean distance from a centroid
    difference = np.linalg.norm(encoded_test_data_point - centroid)
    
    # Define your 'threshold' for anomaly detection
    threshold = 10.0  # Replace with your actual threshold
    
    # Measure the difference and check against the threshold
    if difference > threshold:
        final_prediction = "new_class"
    else:
        final_prediction = np.argmax(original_prediction)  # or any other way to interpret the prediction
else:
    final_prediction = np.argmax(original_prediction)  # or any other way to interpret the prediction

print("Final Prediction:", final_prediction)

```
