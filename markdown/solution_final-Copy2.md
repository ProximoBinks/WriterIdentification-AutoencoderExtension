# IAM Writer Recognition

The goal of the notebook is to use the method explained in the paper [DeepWriter: A Multi-Stream Deep CNN for Text-independent Writer Identification](https://arxiv.org/abs/1606.06472) to identify the writer (author) of a text based on their writing styles. To do so, we'll use the [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database/download-the-iam-handwriting-database).


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

    Number of form-writer pairs: 1539
    [('a01-000u', '000'), ('a01-000x', '001'), ('a01-003', '002'), ('a01-003u', '000'), ('a01-003x', '003')]
    Sample form-writer mappings: [('a01-000u', '000'), ('a01-000x', '001'), ('a01-003', '002'), ('a01-003u', '000'), ('a01-003x', '003')]
    


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

    Top writer IDs: ['000', '150', '151', '152', '153']
    ['000', '150', '151', '152', '153']
    


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

    Number of top forms: 452
    Sample form IDs: ['a01-000u', 'a01-003u', 'a01-007u', 'a01-011u', 'a01-014u']
    ['a01-000u', 'a01-003u', 'a01-007u', 'a01-011u', 'a01-014u']
    


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

    Array lengths: 4909 4909
    


```python
print(f"Checking path: {path_to_files}")
files_found = glob.glob(path_to_files)
print(f"Found {len(files_found)} files.")

print(img_files[0:5])
print(img_targets[0:5])
```

    Checking path: ../data/temp_sentences\*
    Found 4909 files.
    ['../data/temp_sentences\\a01-000u-s00-00.png'
     '../data/temp_sentences\\a01-000u-s00-01.png'
     '../data/temp_sentences\\a01-000u-s00-02.png'
     '../data/temp_sentences\\a01-000u-s00-03.png'
     '../data/temp_sentences\\a01-000u-s01-00.png']
    ['000' '000' '000' '000' '000']
    


```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

for file_name in img_files[:2]:
    img = mpimg.imread(file_name)
    plt.figure(figsize = (10,10))
    plt.imshow(img, cmap ='gray')
```


    
![png](output_11_0.png)
    



    
![png](output_11_1.png)
    



```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoded_img_targets = encoder.fit_transform(img_targets)

print("Writer ID        : ", img_targets[:2])
print("Encoded writer ID: ", encoded_img_targets[:2])
```

    Writer ID        :  ['000' '000']
    Encoded writer ID:  [0 0]
    


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

    Label:  47
    


    
![png](output_16_1.png)
    



```python
images, labels = get_augmented_sample(sample, label, 0.1)
```


```python
print(labels)
print("Num of labels: ", len(labels))
```

    [47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47]
    Num of labels:  158
    


```python
print(len(images))
plt.imshow(images[0], cmap ='gray')
```

    158
    




    <matplotlib.image.AxesImage at 0x1f252ba7eb0>




    
![png](output_19_2.png)
    



```python
plt.imshow(images[1], cmap ='gray')
```




    <matplotlib.image.AxesImage at 0x1f252d32eb0>




    
![png](output_20_1.png)
    



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
    


```python
'''
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
'''
print("Model Trained")
```

    Model Trained
    


```python
# Path to the checkpoint file
#checkpoint_path = './model_checkpoints/old/model_weights.hdf5'
#checkpoint_path = './model_checkpoints/check_20_0.7104.hdf5'
checkpoint_path = './model_checkpoints/check_19_0.7173.hdf5'
model.load_weights(checkpoint_path)
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

    61/61 [==============================] - 14s 236ms/step - loss: 0.6190 - acc: 0.8100
    Accuracy:  0.809961199760437
    


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

    61/61 [==============================] - 14s 235ms/step - loss: 0.6166 - acc: 0.8110
    Accuracy:  0.8110364675521851
    

Model Demo


```python
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
#image_path = '../data/Demo/a01-000u/a01-000u-s00-01.png'
#image_path = '../data/Demo/a01-102u-s01-00.png'
image_path = '../data/Demo/a01-003u-s00-00.png'
img = Image.open(image_path)
plt.imshow(img, cmap='gray')
plt.show()
```


    
![png](output_33_0.png)
    



```python
def preprocess_image(image_path):
    img = Image.open(image_path)
    
    # Compute resize dimensions such that aspect ratio is maintained
    img_width = img.size[0]
    img_height = img.size[1]
    height_fac = CROP_SIZE / img_height
    new_width = int(img_width * height_fac)
    
    # Resize image 
    new_img = img.resize((new_width, CROP_SIZE), Image.LANCZOS)
    
    # Crop the width if it's greater than CROP_SIZE
    if new_width > CROP_SIZE:
        left = (new_width - CROP_SIZE) / 2
        top = 0
        right = (new_width + CROP_SIZE) / 2
        bottom = CROP_SIZE
        new_img = new_img.crop((left, top, right, bottom))
    
    # Ensure the image is exactly CROP_SIZE x CROP_SIZE
    new_img = new_img.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)
    
    # Convert image to numpy array
    img_array = np.array(new_img)
    
    # Reshape to match the input format of the model and normalize
    img_array = img_array.reshape(CROP_SIZE, CROP_SIZE, 1)
    img_array = img_array.astype('float32')
    img_array /= 255
    
    return img_array

#processed_img = preprocess_image('../data/Demo/a01-000u/a01-000u-s00-01.png') #ID 000
#processed_img = preprocess_image('../data/Demo/a01-102u-s01-00.png') #ID 102
processed_img = preprocess_image('../data/Demo/a01-003u-s00-00.png') #ID 003
plt.imshow(processed_img, cmap='gray')
plt.show()
```


    
![png](output_34_0.png)
    



```python
# Expand dimensions to fit the model's input shape
input_data = np.expand_dims(processed_img, axis=0)

# Predict
prediction = model.predict(input_data)
predicted_label = np.argmax(prediction, axis=1)
print(prediction)
```

    1/1 [==============================] - 0s 75ms/step
    [[9.9999619e-01 4.0172651e-24 1.7177088e-14 1.8783735e-21 3.7083275e-08
      1.1923955e-15 1.5513027e-14 2.1188125e-20 6.4238079e-17 1.3183204e-17
      2.8896983e-25 6.9812262e-10 8.5474301e-21 3.8636048e-15 1.9553059e-24
      8.1143961e-14 3.4639516e-20 2.8565239e-13 1.4314926e-10 4.9457231e-28
      3.4858003e-18 4.7144499e-23 1.6837527e-18 3.9063453e-15 2.6060835e-13
      2.8416340e-13 1.2295737e-10 2.8346755e-11 3.6610233e-19 1.1046351e-13
      1.2511692e-09 1.2614773e-17 9.3268735e-09 1.6793053e-13 1.2569057e-15
      1.4019707e-23 6.8190205e-15 6.2034876e-22 3.6339291e-06 1.0022217e-13
      3.8402513e-25 8.3536301e-22 8.5678304e-14 7.0386652e-10 2.0368860e-07
      1.1223736e-12 1.2737007e-18 2.3477511e-16 2.0996067e-20 3.8621706e-10]]
    


```python
# Decode the predicted label to get the writer's ID
writer_id = encoder.inverse_transform(predicted_label)
print(f"Predicted Writer ID: {writer_id[0]}")
```

    Predicted Writer ID: 000
    

### Train the Autoencoder


```python
from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

# Define the shape of the input images
input_shape = (128, 128, 1)

# Encoder
input_img = Input(shape=input_shape)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Full autoencoder
autoencoder = Model(input_img, decoded)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(images, images, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

# Save the model
autoencoder.save('autoencoder_model.h5')

```

    Epoch 1/50
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[35], line 28
         25 autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
         27 # Train the autoencoder
    ---> 28 autoencoder.fit(images, images, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)
         30 # Save the model
         31 autoencoder.save('autoencoder_model.h5')
    

    File ~\anaconda3\lib\site-packages\keras\utils\traceback_utils.py:67, in filter_traceback.<locals>.error_handler(*args, **kwargs)
         65 except Exception as e:  # pylint: disable=broad-except
         66   filtered_tb = _process_traceback_frames(e.__traceback__)
    ---> 67   raise e.with_traceback(filtered_tb) from None
         68 finally:
         69   del filtered_tb
    

    File ~\AppData\Local\Temp\__autograph_generated_filely5rk_fz.py:15, in outer_factory.<locals>.inner_factory.<locals>.tf__train_function(iterator)
         13 try:
         14     do_return = True
    ---> 15     retval_ = ag__.converted_call(ag__.ld(step_function), (ag__.ld(self), ag__.ld(iterator)), None, fscope)
         16 except:
         17     do_return = False
    

    ValueError: in user code:
    
        File "C:\Users\User\anaconda3\lib\site-packages\keras\engine\training.py", line 1051, in train_function  *
            return step_function(self, iterator)
        File "C:\Users\User\anaconda3\lib\site-packages\keras\engine\training.py", line 1040, in step_function  **
            outputs = model.distribute_strategy.run(run_step, args=(data,))
        File "C:\Users\User\anaconda3\lib\site-packages\keras\engine\training.py", line 1030, in run_step  **
            outputs = model.train_step(data)
        File "C:\Users\User\anaconda3\lib\site-packages\keras\engine\training.py", line 889, in train_step
            y_pred = self(x, training=True)
        File "C:\Users\User\anaconda3\lib\site-packages\keras\utils\traceback_utils.py", line 67, in error_handler
            raise e.with_traceback(filtered_tb) from None
        File "C:\Users\User\anaconda3\lib\site-packages\keras\engine\input_spec.py", line 200, in assert_input_compatibility
            raise ValueError(f'Layer "{layer_name}" expects {len(input_spec)} input(s),'
    
        ValueError: Layer "model" expects 1 input(s), but it received 158 input tensors. Inputs received: [<tf.Tensor 'IteratorGetNext:0' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:1' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:2' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:3' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:4' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:5' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:6' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:7' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:8' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:9' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:10' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:11' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:12' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:13' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:14' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:15' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:16' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:17' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:18' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:19' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:20' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:21' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:22' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:23' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:24' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:25' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:26' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:27' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:28' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:29' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:30' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:31' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:32' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:33' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:34' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:35' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:36' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:37' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:38' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:39' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:40' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:41' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:42' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:43' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:44' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:45' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:46' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:47' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:48' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:49' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:50' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:51' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:52' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:53' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:54' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:55' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:56' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:57' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:58' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:59' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:60' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:61' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:62' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:63' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:64' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:65' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:66' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:67' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:68' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:69' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:70' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:71' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:72' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:73' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:74' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:75' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:76' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:77' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:78' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:79' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:80' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:81' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:82' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:83' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:84' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:85' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:86' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:87' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:88' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:89' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:90' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:91' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:92' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:93' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:94' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:95' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:96' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:97' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:98' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:99' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:100' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:101' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:102' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:103' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:104' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:105' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:106' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:107' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:108' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:109' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:110' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:111' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:112' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:113' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:114' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:115' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:116' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:117' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:118' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:119' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:120' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:121' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:122' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:123' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:124' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:125' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:126' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:127' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:128' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:129' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:130' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:131' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:132' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:133' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:134' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:135' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:136' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:137' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:138' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:139' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:140' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:141' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:142' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:143' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:144' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:145' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:146' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:147' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:148' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:149' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:150' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:151' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:152' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:153' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:154' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:155' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:156' shape=(None, 113) dtype=uint8>, <tf.Tensor 'IteratorGetNext:157' shape=(None, 113) dtype=uint8>]
    



```python
from keras.models import load_model
from sklearn.metrics import mean_squared_error

# Load the trained autoencoder
autoencoder = load_model('autoencoder_model.h5')

# Extract the encoder part
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[-7].output)

# Function to get latent representations
def get_latent_representations(images):
    return encoder.predict(images)

# Calculate the centroid of the latent representations of the training data
latent_representations = get_latent_representations(images)
centroid = np.mean(latent_representations, axis=0)

# For a new image:
new_image = preprocess_image('./data/Demo/a01-043u/a01-043u-s00-00.png')  # adjust path
new_image = np.expand_dims(new_image, axis=0)
reconstructed_image = autoencoder.predict(new_image)

# Calculate reconstruction error
error = mean_squared_error(new_image.flatten(), reconstructed_image.flatten())
print(f"Reconstruction error: {error}")

# Get latent representation for the new image
new_latent_representation = get_latent_representations(new_image)

# Calculate distance from the centroid
distance = np.linalg.norm(new_latent_representation - centroid)
print(f"Distance from centroid: {distance}")

# Decide if it's a new writer (you might need to adjust the threshold based on your observations)
threshold = 0.5
if distance > threshold:
    print("It might be a new writer!")
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

### Extract the Encoder


```python
# Encoder model for feature extraction
encoder = Model(input_img, encoded)
```


```python
from collections import defaultdict
import numpy as np

writer_centroids = defaultdict(list)

# Loop through each writer's images
for writer_id in top_writers:  # Assuming top_writers contains the writer IDs
    writer_images = []  # Collect all images for this writer
    for form_id, writer in form_writer.items():
        if writer == writer_id:
            image_path = os.path.join(temp_sentences_path, form_id + '.png')  # Adjust as needed
            image = Image.open("../data/a01-043u-s04-03.png")
            image = image.resize((113, 113))
            writer_images.append(np.asarray(image))
            
    writer_images = np.array(writer_images).reshape(len(writer_images), 113, 113, 1)
    encoded_images = encoder.predict(writer_images)
    centroid = np.mean(encoded_images, axis=0)
    writer_centroids[writer_id] = centroid

```


```python
writer_thresholds = {}

for writer_id, centroid in writer_centroids.items():
    writer_images = []  # Collect all images for this writer
    for form_id, writer in form_writer.items():
        if writer == writer_id:
            image_path = os.path.join(temp_sentences_path, form_id + '.png')  # Adjust as needed
            image = Image.open("../data/a01-043u-s04-03.png")
            image = image.resize((113, 113))
            writer_images.append(np.asarray(image))
            
    writer_images = np.array(writer_images).reshape(len(writer_images), 113, 113, 1)
    encoded_images = encoder.predict(writer_images)
    distances = [np.linalg.norm(encoded - centroid) for encoded in encoded_images]
    avg_distance = np.mean(distances)
    writer_thresholds[writer_id] = avg_distance * 1.2  # Adding some margin

```


```python
def identify_writer(new_image, writer_centroids, writer_thresholds):
    # Preprocess the new image and encode it
    new_image = Image.open(new_image).resize((113, 113))
    new_image = np.asarray(new_image).reshape(1, 113, 113, 1)
    encoded_image = encoder.predict(new_image)

    for writer_id, centroid in writer_centroids.items():
        distance = np.linalg.norm(encoded_image - centroid)
        if distance < writer_thresholds[writer_id]:
            print(f"Writer identified as {writer_id}")
            return writer_id

    print("This is a new writer.")
    return None

```


```python

```

### Anomaly Detection for New Class


```python
# Assume you have some 'normal_data_points' and one 'new_data_point'
normal_data_points = np.random.rand(100, 113, 113, 1)  # Replace with your actual normal data
new_data_point = np.random.rand(1, 113, 113, 1)  # Replace with your actual new data point

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

### Add new class


```python
from PIL import Image
import numpy as np

# Step 1: Read the image
img_path = 'path/to/your/test/image.png'  # Replace with the actual path
img = Image.open(img_path)

# Step 2: Resize and preprocess
img = img.resize((CROP_SIZE, CROP_SIZE))  # Replace CROP_SIZE with the actual size
img_array = np.asarray(img)

# Step 3: Expand dimensions to match the input shape
test_data_point = np.expand_dims(img_array, axis=0)  # Makes it (1, CROP_SIZE, CROP_SIZE, num_channels)

# If your images are grayscale and your model expects a single channel image, you might also need to reshape:
test_data_point = test_data_point.reshape(test_data_point.shape[0], CROP_SIZE, CROP_SIZE, 1)

# Normalize the data if your model expects that
test_data_point = test_data_point.astype('float32') / 255.0

```

### Integrate with Original Classifier


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


```python

```
