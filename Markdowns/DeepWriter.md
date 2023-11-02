```python
import cv2
from glob import glob
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from tqdm.notebook import tqdm

import tensorflow as tf
import PIL.ImageOps

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Add
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
```


```python

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def slicer(images, names, segment = 80, timeAxis=False):
    half = segment//4
    
    retImg = []
    retClass = []
    totbit = segment*segment
    
    for img, name in zip(images, names):
        if timeAxis == True:
            timeImage = []

        for i in range(0, img.shape[1], half):
            if i+half*3 > img.shape[1]:
                continue
            
            tmp = img[:, i:i+segment]
            tmp = np.pad(tmp, ((0, 0), (0, segment-tmp.shape[1])), 'constant', 
                         constant_values=0)

            if timeAxis == True:
                timeImage.append(tmp)
            else:
                retImg.append(tmp)
                retClass.append(name)

        if timeAxis == True:
            retImg.append(np.array(timeImage))
            retClass.append(name)
    
    X = np.array(retImg)
    y = np.array(retClass)
    
    return X, y

# Load data
def loadData(perClassData=None, h=80):
    '''
    Give the directory of dataset in the glob function
    Generate target name/identity in name variable
    '''
    imgFiles = glob("./data/temp_sentences/*.png")
    print(len(imgFiles), 'images found.')

    ImageArray = []
    Names = []

    for imgFile in tqdm(imgFiles):  # <-- This line was corrected
        fileName = (imgFile.split('/')[-1]).split('.')[0]
        name = fileName.split('_')[0]                           # Target Class

        img = Image.open(imgFile)
        img = PIL.ImageOps.invert(img)
        image = image_resize(np.array(img, dtype=np.uint8), height=h)
        
        if image.ndim > 2:
            continue

        image = image / 255
        ImageArray.append(image)
        Names.append(name)
    
    print('Total Unique Classes', len(np.unique(Names)))
    return ImageArray, Names  

```


```python
X, y = loadData(60, h=113)
```

    4909 images found.
    


      0%|          | 0/4909 [00:00<?, ?it/s]


    Total Unique Classes 1
    

X is the black and white word image of shape (row, cols, 1).

row and col doesn't have to be same.

y is the target class.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                    random_state=40)

# XX_test, yy_test is the multiple (113, 113) segmented images of word line
XX_test, yy_test = slicer(X_test, y_test, segment=113, timeAxis=True)

# Set timeAxis=False for a lower dimention
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[4], line 7
          3 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
          4                                                     random_state=40)
          6 # XX_test, yy_test is the multiple (113, 113) segmented images of word line
    ----> 7 XX_test, yy_test = slicer(X_test, y_test, segment=113, timeAxis=True)
    

    Cell In[2], line 61, in slicer(images, names, segment, timeAxis)
         58         retImg.append(np.array(timeImage))
         59         retClass.append(name)
    ---> 61 X = np.array(retImg)
         62 y = np.array(retClass)
         64 return X, y
    

    ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (982,) + inhomogeneous part.



```python
from sklearn.preprocessing import OneHotEncoder
OHE = OneHotEncoder().fit(np.array(y).reshape(-1, 1))

y_train = np.array(y_train)
y_test = np.array(y_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

y_train_OHE = OHE.fit_transform(y_train).toarray()
y_test_OHE = OHE.transform(y_test).toarray()
```


```python
def deepWriter(input_shape, classes):
    # Two different input patches
    patch_1 = Input(shape=input_shape)
    patch_2 = Input(shape=input_shape)

    # Convolution_1 shares the same weight
    conv1 = Conv2D(96, kernel_size=5, strides=2, activation='relu')
    out1 = conv1(patch_1)
    out2 = conv1(patch_2)

    # MaxPooling
    MP = MaxPooling2D(3, strides=2)
    out1 = MP(out1)
    out2 = MP(out2)

    # Convolution_2 shares the same weight
    conv2 = Conv2D(256, kernel_size=3, activation='relu')
    out1 = conv2(out1)
    out2 = conv2(out2)

    # MaxPooling
    out1 = MP(out1)
    out2 = MP(out2)

    # Convolution_3 shares the same weight
    conv3 = Conv2D(384, kernel_size=3, activation='relu')
    out1 = conv3(out1)
    out2 = conv3(out2)

    # Convolution_4 shares the same weight
    conv4 = Conv2D(384, kernel_size=3, activation='relu')
    out1 = conv4(out1)
    out2 = conv4(out2)

    # Convolution_5 shares the same weight
    conv5 = Conv2D(256, kernel_size=3, activation='relu')
    out1 = conv5(out1)
    out2 = conv5(out2)

    # MaxPooling
    out1 = MP(out1)
    out2 = MP(out2)

    # Flatten
    flat = Flatten()
    out1 = flat(out1)
    out2 = flat(out2)

    # Fully Connected Layer (FC6)
    FC6 = Dense(1024)
    out1 = FC6(out1)
    out2 = FC6(out2)

    # Dropout of 0.5
    out1 = Dropout(0.5)(out1)
    out2 = Dropout(0.5)(out2)

    # Fully Conneted Layer (FC7)
    FC7 = Dense(1024)
    out1 = FC7(out1)
    out2 = FC7(out2)

    # Dropout of 0.5
    out1 = Dropout(0.5)(out1)
    out2 = Dropout(0.5)(out2)

    # Summation of two outputs
    out = Add()([out1, out2])

    # Softmax layer
    out = Dense(classes, activation='softmax')(out)

    # Make model and compile
    model = Model(inputs=[patch_1, patch_2], outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['acc'])

    return model


def halfDeepWriter(input_shape, classes, frac=1):
    patch_1 = Input(shape=input_shape)

    out1 = Conv2D(int(96*frac), kernel_size=5, strides=2, activation='relu')(patch_1)
    out1 = MaxPooling2D(3, strides=2)(out1)

    out1 = Conv2D(int(256*frac), kernel_size=3, activation='relu')(out1)
    out1 = MaxPooling2D(3, strides=2)(out1)

    out1 = Conv2D(int(384*frac), kernel_size=3, activation='relu')(out1)
    out1 = Conv2D(int(384*frac), kernel_size=3, activation='relu')(out1)
    out1 = Conv2D(int(256*frac), kernel_size=3, activation='relu')(out1)
    out1 = MaxPooling2D(3, strides=2)(out1)

    out1 = Flatten()(out1)
    out1 = Dense(int(1024*frac), activation='relu')(out1)
    out1 = Dropout(0.5)(out1)

    out1 = Dense(int(1024*frac), activation='relu')(out1)
    out1 = Dropout(0.5)(out1)

    out1 = Dense(classes, activation='softmax')(out1)

    model = Model(inputs=patch_1, outputs=out1)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['acc'])

    return model
```


```python
# Random image strip image generator of DeepWriter's image stripping strategy

class dataGeneratorDeepWriter(tensorflow.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, shuffle=True, w=80):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.inputX = X
        self.inputY = y
        self.w = w
        self.h = self.inputX[0].shape[0]
        self.total = len(X)
        self.indexes = np.arange(self.total)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.total / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batchIndexes):
        'Generates data containing batch_size samples' # X : (2, n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, 2, self.h, self.w))
        y = np.empty((self.batch_size, self.inputY.shape[-1]), dtype=int)
        
        # Generate data
        for i, ID in enumerate(batchIndexes):
            # Black Image
            tmpImg = np.zeros((self.h, self.w))
            
            # Starting column position
            y_pos1, y_pos2 = map(int, (np.random.randint(low=0, 
                        high=max(self.inputX[ID].shape[1]-self.w//3, 1),
                        size=2)))
            
            # Placing Image in black image
            tmpImg1 = (self.inputX[ID])[:, y_pos1:y_pos1+self.w]
            tmpImg2 = (self.inputX[ID])[:, y_pos2:y_pos2+self.w]

            # Placing Image in output
            X[i, 0, 0:tmpImg1.shape[0], 0:tmpImg1.shape[1]] = tmpImg1
            X[i, 1, 0:tmpImg2.shape[0], 0:tmpImg2.shape[1]] = tmpImg2
            
            # Store class
            y[i] = self.inputY[ID]

        X = X[:, :, :, :, np.newaxis]
        return [X[:, 0, :, :], X[:, 1, :, :]], y


class dataGeneratorHalfDeepWriter(tensorflow.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, shuffle=True, w=80):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.inputX = X
        self.inputY = y
        self.w = w
        self.h = self.inputX[0].shape[0]
        self.total = len(X)
        self.indexes = np.arange(self.total)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.total / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batchIndexes):
        'Generates data containing batch_size samples' # X : (2, n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, self.h, self.w))
        y = np.empty((self.batch_size, self.inputY.shape[-1]), dtype=int)
        
        # Generate data
        for i, ID in enumerate(batchIndexes):
            # Black Image
            tmpImg = np.zeros((self.h, self.w))
            
            # Starting column position
            y_pos1 = int(np.random.randint(low=0, 
                        high=max(self.inputX[ID].shape[1]-self.w//3, 1),
                        size=1))
            
            # Placing Image in black image
            tmpImg1 = (self.inputX[ID])[:, y_pos1:y_pos1+self.w]

            # Placing Image in output
            X[i, 0:tmpImg1.shape[0], 0:tmpImg1.shape[1]] = tmpImg1
            
            # Store class
            y[i] = self.inputY[ID]

        X = X[:, :, :, np.newaxis]
        return X, y

```


```python
model = halfDeepWriter((113, 113, 1), 54, )
model.summary()
#model.load_weights('/content/best.hdf5')
```


```python
train_gen = dataGeneratorHalfDeepWriter(X_train, y_train_OHE, batch_size=128, w=113)
test_gen = dataGeneratorHalfDeepWriter(X_test, y_test_OHE, batch_size=128, w=113)

hist = model.fit(train_gen, validation_data=test_gen, epochs=3000, 
                 callbacks=[ ModelCheckpoint(filepath='/content/best.hdf5',
                             save_best_only=True, monitor='acc', mode='max',
                            ), ])
```


```python
# Calculating word-level accuracy

from sklearn.metrics import accuracy_score

y_pred = []
y_true = []

for batch, tar in zip(XX_test, yy_test_OHE):
    if batch.shape[0] <= 0:
        continue
    batch = batch[:, :, :, np.newaxis]
    y_pred.append(np.argmax(np.sum(model.predict(batch), axis=0), axis=0))
    y_true.append(np.argmax(tar))

accuracy_score(y_true, y_pred)
```
