## Imports and Constants


```python
# Import necessary libraries
import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten, Dropout, Lambda, Activation
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from PIL import Image
from functools import reduce
from itertools import islice
from collections import Counter

# Constants
CROP_SIZE = 113
NUM_LABELS = 50
BATCH_SIZE = 16
```

## Create Autoencoder


```python
# Function to create the autoencoder
def create_autoencoder():
    input_img = Input(shape=(CROP_SIZE, CROP_SIZE, 1))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(1, activation='sigmoid')(decoded)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder
```

## Extract Features Using Autoencoder


```python
# Function to extract features using the autoencoder
def extract_features(encoder_model, data):
    data_features = encoder_model.predict(data.reshape((len(data), CROP_SIZE * CROP_SIZE)))
    return data_features
```

## Create Writer Identification Model


```python
# Function to create and compile the writer identification model
def create_writer_identification_model():
    model = Sequential()

    # Define network input shape
    model.add(ZeroPadding2D((1, 1), input_shape=(CROP_SIZE, CROP_SIZE, 1)))
    model.add(Lambda(resize_image))

    # CNN model
    model.add(Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same', name='conv1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1'))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2'))

    model.add(Convolution2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv3'))
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

    return model
```

## Generate Data Batches


```python
# Function to generate data batches
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

## Resize Image


```python
# Function to resize images
def resize_image(img):
    size = round(CROP_SIZE / 2)
    return tf.image.resize(img, [size, size])
```

## Load and preprocess the data


```python
import os
import glob
import shutil
import numpy as np
from itertools import islice
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from collections import Counter

def load_and_preprocess_data():
    # Create a dictionary to store each form ID and its writer
    form_writer = {}
    forms_file_path = "../data/forms.txt"
    
    with open(forms_file_path) as f:
        for line in islice(f, 16, None):
            line_list = line.split(' ')
            form_id = line_list[0]
            writer = line_list[1]
            form_writer[form_id] = writer

    # Select the 50 most common writers
    num_writers = 50
    writers_counter = Counter(form_writer.values())
    top_writers = [writer_id for writer_id, _ in writers_counter.most_common(num_writers)]

    # Create a temp directory containing only the selected sentences
    temp_sentences_path = "../data/temp_sentences"
    if not os.path.exists(temp_sentences_path):
        os.makedirs(temp_sentences_path)

    original_sentences_path = os.path.join("../data/sentences", "*", "*", "*.png")

    for file_path in glob.glob(original_sentences_path):
        image_name = file_path.split(os.path.sep)[-1]
        form_id = image_name.split('-')[0] + '-' + image_name.split('-')[1]

        if form_id in top_forms:
            try:
                shutil.copy(file_path, os.path.join(temp_sentences_path, image_name))
            except Exception as e:
                print(f"Failed to copy {file_path}. Error: {e}")

    # Create lists of file inputs (a form) and their respective targets (a writer id)
    img_files = []
    img_targets = []

    path_to_files = os.path.join(temp_sentences_path, "*", "*", "*.png")
    for file_path in glob.glob(path_to_files):
        img_files.append(file_path)
        img_targets.append(form_writer[file_path.split(os.path.sep)[-2]])

    # Encode target values
    encoder = LabelEncoder()
    img_targets_encoded = encoder.fit_transform(img_targets)

    # Normalize the pixel values and convert images to arrays
    img_data = []
    for img_path in img_files:
        img = Image.open(img_path).convert('L')
        img_data.append(np.array(img) / 255.0)

    # Split the dataset into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(img_data, img_targets_encoded, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test
```

## Function to train the writer identification model


```python
def train_writer_identification_model(X_train, y_train, X_val, y_val):
    writer_identification_model = create_writer_identification_model()

    # Define model checkpoint to save the best model
    checkpoint = ModelCheckpoint("best_model.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # Train the model
    history = writer_identification_model.fit(
        generate_data(X_train, y_train, BATCH_SIZE, sample_ratio=0.5),
        steps_per_epoch=len(X_train) / BATCH_SIZE,
        validation_data=(X_val, to_categorical(y_val, NUM_LABELS)),
        epochs=20,
        callbacks=[checkpoint],
        verbose=1
    )

    return writer_identification_model, history
```

## Main code


```python
if __name__ == "__main__":
    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()

    # Create and train the autoencoder
    autoencoder = create_autoencoder()
    autoencoder.fit(X_train, X_train, epochs=10, batch_size=32)

    # Extract features using the autoencoder
    encoder_model = Model(inputs=autoencoder.input, outputs=autoencoder.layers[3].output)
    X_train_features = extract_features(encoder_model, X_train)
    X_val_features = extract_features(encoder_model, X_val)
    X_test_features = extract_features(encoder_model, X_test)

    # Train the writer identification model
    writer_identification_model, history = train_writer_identification_model(X_train_features, y_train, X_val_features, y_val)

    # Evaluate the model on the test set
    test_loss, test_accuracy = writer_identification_model.evaluate(X_test_features, to_categorical(y_test, NUM_LABELS))
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [9], in <cell line: 1>()
          1 if __name__ == "__main__":
          2     # Load and preprocess data
    ----> 3     X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()
          5     # Create and train the autoencoder
          6     autoencoder = create_autoencoder()
    

    Input In [7], in load_and_preprocess_data()
         36 image_name = file_path.split(os.path.sep)[-1]
         37 form_id = image_name.split('-')[0] + '-' + image_name.split('-')[1]
    ---> 39 if form_id in top_forms:
         40     try:
         41         shutil.copy(file_path, os.path.join(temp_sentences_path, image_name))
    

    NameError: name 'top_forms' is not defined

