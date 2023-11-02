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

def slicer(images, names, segment=80, timeAxis=False):
    half = segment // 4
    retImg = []
    retClass = []
    
    for img, name in zip(images, names):
        if timeAxis:
            timeImage = []

        for i in range(0, img.shape[1], half):
            if i + half * 3 > img.shape[1]:
                continue
            
            tmp = img[:, i:i + segment]
            tmp = np.pad(tmp, ((0, 0), (0, segment - tmp.shape[1])), 'constant', constant_values=0)

            if timeAxis:
                timeImage.append(tmp)
            else:
                retImg.append(tmp)
                retClass.append(name)

        if timeAxis and len(timeImage) > 0:   # Check if timeImage has content
            retImg.append(np.stack(timeImage))
            retClass.append(name)

    return retImg, retClass


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
        name = fileName.split('-')[0]                           # Target Class

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


    Total Unique Classes 25
    

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
import tensorflow
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
model = halfDeepWriter((113, 113, 1), 25, )
model.summary()
#model.load_weights('/content/best.hdf5')
```

    Model: "model_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_2 (InputLayer)        [(None, 113, 113, 1)]     0         
                                                                     
     conv2d_5 (Conv2D)           (None, 55, 55, 96)        2496      
                                                                     
     max_pooling2d_3 (MaxPoolin  (None, 27, 27, 96)        0         
     g2D)                                                            
                                                                     
     conv2d_6 (Conv2D)           (None, 25, 25, 256)       221440    
                                                                     
     max_pooling2d_4 (MaxPoolin  (None, 12, 12, 256)       0         
     g2D)                                                            
                                                                     
     conv2d_7 (Conv2D)           (None, 10, 10, 384)       885120    
                                                                     
     conv2d_8 (Conv2D)           (None, 8, 8, 384)         1327488   
                                                                     
     conv2d_9 (Conv2D)           (None, 6, 6, 256)         884992    
                                                                     
     max_pooling2d_5 (MaxPoolin  (None, 2, 2, 256)         0         
     g2D)                                                            
                                                                     
     flatten_1 (Flatten)         (None, 1024)              0         
                                                                     
     dense_3 (Dense)             (None, 1024)              1049600   
                                                                     
     dropout_2 (Dropout)         (None, 1024)              0         
                                                                     
     dense_4 (Dense)             (None, 1024)              1049600   
                                                                     
     dropout_3 (Dropout)         (None, 1024)              0         
                                                                     
     dense_5 (Dense)             (None, 25)                25625     
                                                                     
    =================================================================
    Total params: 5446361 (20.78 MB)
    Trainable params: 5446361 (20.78 MB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    


```python
train_gen = dataGeneratorHalfDeepWriter(X_train, y_train_OHE, batch_size=128, w=113)
test_gen = dataGeneratorHalfDeepWriter(X_test, y_test_OHE, batch_size=128, w=113)

hist = model.fit(train_gen, validation_data=test_gen, epochs=50, 
                 callbacks=[ ModelCheckpoint(filepath='/content/best.hdf5',
                             save_best_only=True, monitor='acc', mode='max',
                            ), ])
```

    Epoch 1/50
    30/30 [==============================] - 31s 984ms/step - loss: 3.2083 - acc: 0.2148 - val_loss: 2.5777 - val_acc: 0.3047
    Epoch 2/50
    

    C:\Users\User\AppData\Roaming\Python\Python311\site-packages\keras\src\engine\training.py:3079: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
      saving_api.save_model(
    

    30/30 [==============================] - 30s 984ms/step - loss: 2.5127 - acc: 0.3250 - val_loss: 2.5169 - val_acc: 0.3058
    Epoch 3/50
    30/30 [==============================] - 30s 1s/step - loss: 2.4821 - acc: 0.3250 - val_loss: 2.5034 - val_acc: 0.3125
    Epoch 4/50
    30/30 [==============================] - 29s 975ms/step - loss: 2.4818 - acc: 0.3242 - val_loss: 2.5443 - val_acc: 0.3025
    Epoch 5/50
    30/30 [==============================] - 29s 962ms/step - loss: 2.4665 - acc: 0.3273 - val_loss: 2.5175 - val_acc: 0.3036
    Epoch 6/50
    30/30 [==============================] - 30s 1s/step - loss: 2.4521 - acc: 0.3245 - val_loss: 2.4448 - val_acc: 0.3013
    Epoch 7/50
    30/30 [==============================] - 29s 966ms/step - loss: 2.3911 - acc: 0.3305 - val_loss: 2.4179 - val_acc: 0.3025
    Epoch 8/50
    30/30 [==============================] - 29s 977ms/step - loss: 2.3335 - acc: 0.3383 - val_loss: 2.3002 - val_acc: 0.3259
    Epoch 9/50
    30/30 [==============================] - 30s 1s/step - loss: 2.2556 - acc: 0.3562 - val_loss: 2.3159 - val_acc: 0.3125
    Epoch 10/50
    30/30 [==============================] - 30s 996ms/step - loss: 2.2582 - acc: 0.3536 - val_loss: 2.2518 - val_acc: 0.3438
    Epoch 11/50
    30/30 [==============================] - 30s 987ms/step - loss: 2.2315 - acc: 0.3607 - val_loss: 2.2661 - val_acc: 0.3382
    Epoch 12/50
    30/30 [==============================] - 29s 973ms/step - loss: 2.2078 - acc: 0.3552 - val_loss: 2.1979 - val_acc: 0.3404
    Epoch 13/50
    30/30 [==============================] - 30s 987ms/step - loss: 2.1654 - acc: 0.3643 - val_loss: 2.1498 - val_acc: 0.3627
    Epoch 14/50
    30/30 [==============================] - 29s 977ms/step - loss: 2.0943 - acc: 0.3862 - val_loss: 2.0652 - val_acc: 0.3873
    Epoch 15/50
    30/30 [==============================] - 30s 1s/step - loss: 2.0090 - acc: 0.4378 - val_loss: 1.9640 - val_acc: 0.4353
    Epoch 16/50
    30/30 [==============================] - 31s 1s/step - loss: 1.9688 - acc: 0.4430 - val_loss: 2.0135 - val_acc: 0.4085
    Epoch 17/50
    30/30 [==============================] - 30s 999ms/step - loss: 1.9217 - acc: 0.4698 - val_loss: 1.8181 - val_acc: 0.4844
    Epoch 18/50
    30/30 [==============================] - 30s 985ms/step - loss: 1.8432 - acc: 0.4732 - val_loss: 1.7906 - val_acc: 0.4888
    Epoch 19/50
    30/30 [==============================] - 30s 987ms/step - loss: 1.7765 - acc: 0.4932 - val_loss: 1.7315 - val_acc: 0.4888
    Epoch 20/50
    30/30 [==============================] - 30s 1s/step - loss: 1.7690 - acc: 0.4914 - val_loss: 1.6786 - val_acc: 0.5112
    Epoch 21/50
    30/30 [==============================] - 31s 1s/step - loss: 1.7703 - acc: 0.4961 - val_loss: 1.7815 - val_acc: 0.4609
    Epoch 22/50
    30/30 [==============================] - 30s 987ms/step - loss: 1.7033 - acc: 0.5039 - val_loss: 1.6230 - val_acc: 0.5100
    Epoch 23/50
    30/30 [==============================] - 30s 1s/step - loss: 1.6361 - acc: 0.5096 - val_loss: 1.6444 - val_acc: 0.4989
    Epoch 24/50
    30/30 [==============================] - 29s 979ms/step - loss: 1.6427 - acc: 0.5182 - val_loss: 1.6596 - val_acc: 0.4978
    Epoch 25/50
    30/30 [==============================] - 30s 997ms/step - loss: 1.6253 - acc: 0.5185 - val_loss: 1.5200 - val_acc: 0.5257
    Epoch 26/50
    30/30 [==============================] - 31s 1s/step - loss: 1.5664 - acc: 0.5341 - val_loss: 1.5232 - val_acc: 0.5312
    Epoch 27/50
    30/30 [==============================] - 30s 996ms/step - loss: 1.5616 - acc: 0.5393 - val_loss: 1.5849 - val_acc: 0.5234
    Epoch 28/50
    30/30 [==============================] - 30s 996ms/step - loss: 1.5411 - acc: 0.5339 - val_loss: 1.5241 - val_acc: 0.5234
    Epoch 29/50
    30/30 [==============================] - 30s 996ms/step - loss: 1.5226 - acc: 0.5359 - val_loss: 1.4838 - val_acc: 0.5335
    Epoch 30/50
    30/30 [==============================] - 30s 1s/step - loss: 1.4941 - acc: 0.5417 - val_loss: 1.4904 - val_acc: 0.5324
    Epoch 31/50
    30/30 [==============================] - 31s 1s/step - loss: 1.4720 - acc: 0.5443 - val_loss: 1.4471 - val_acc: 0.5658
    Epoch 32/50
    30/30 [==============================] - 30s 1s/step - loss: 1.4583 - acc: 0.5513 - val_loss: 1.3661 - val_acc: 0.5759
    Epoch 33/50
    30/30 [==============================] - 30s 1s/step - loss: 1.3810 - acc: 0.5656 - val_loss: 1.3592 - val_acc: 0.5614
    Epoch 34/50
    30/30 [==============================] - 30s 1s/step - loss: 1.4021 - acc: 0.5617 - val_loss: 1.2720 - val_acc: 0.5982
    Epoch 35/50
    30/30 [==============================] - 31s 1s/step - loss: 1.3953 - acc: 0.5651 - val_loss: 1.3309 - val_acc: 0.5781
    Epoch 36/50
    30/30 [==============================] - 30s 989ms/step - loss: 1.3484 - acc: 0.5708 - val_loss: 1.3430 - val_acc: 0.5804
    Epoch 37/50
    30/30 [==============================] - 30s 1s/step - loss: 1.3762 - acc: 0.5703 - val_loss: 1.3732 - val_acc: 0.5804
    Epoch 38/50
    30/30 [==============================] - 30s 1s/step - loss: 1.3609 - acc: 0.5771 - val_loss: 1.2896 - val_acc: 0.5893
    Epoch 39/50
    30/30 [==============================] - 31s 1s/step - loss: 1.3056 - acc: 0.5846 - val_loss: 1.2016 - val_acc: 0.6127
    Epoch 40/50
    30/30 [==============================] - 31s 1s/step - loss: 1.3276 - acc: 0.5753 - val_loss: 1.2651 - val_acc: 0.5792
    Epoch 41/50
    30/30 [==============================] - 30s 996ms/step - loss: 1.2514 - acc: 0.6003 - val_loss: 1.2519 - val_acc: 0.5904
    Epoch 42/50
    30/30 [==============================] - 30s 1s/step - loss: 1.2615 - acc: 0.5911 - val_loss: 1.2467 - val_acc: 0.5926
    Epoch 43/50
    30/30 [==============================] - 31s 1s/step - loss: 1.2761 - acc: 0.5979 - val_loss: 1.2512 - val_acc: 0.6027
    Epoch 44/50
    30/30 [==============================] - 29s 974ms/step - loss: 1.2756 - acc: 0.5987 - val_loss: 1.2831 - val_acc: 0.5692
    Epoch 45/50
    30/30 [==============================] - 29s 977ms/step - loss: 1.2084 - acc: 0.6052 - val_loss: 1.2389 - val_acc: 0.5904
    Epoch 46/50
    30/30 [==============================] - 30s 984ms/step - loss: 1.2194 - acc: 0.6099 - val_loss: 1.2520 - val_acc: 0.6094
    Epoch 47/50
    30/30 [==============================] - 30s 1s/step - loss: 1.1691 - acc: 0.6279 - val_loss: 1.1446 - val_acc: 0.6194
    Epoch 48/50
    30/30 [==============================] - 30s 995ms/step - loss: 1.1751 - acc: 0.6219 - val_loss: 1.1121 - val_acc: 0.6362
    Epoch 49/50
    30/30 [==============================] - 30s 1s/step - loss: 1.1445 - acc: 0.6240 - val_loss: 1.2369 - val_acc: 0.5915
    Epoch 50/50
    30/30 [==============================] - 30s 1s/step - loss: 1.1496 - acc: 0.6310 - val_loss: 1.0699 - val_acc: 0.6529
    


```python
model.save("my_model.h5")
```


```python
# Calculating word-level accuracy

from sklearn.metrics import accuracy_score

y_pred = []
y_true = []

for batch, tar in zip(XX_test, y_test_OHE):
    if batch.shape[0] <= 0:
        continue
    batch = batch[:, :, :, np.newaxis]
    y_pred.append(np.argmax(np.sum(model.predict(batch), axis=0), axis=0))
    y_true.append(np.argmax(tar))

accuracy_score(y_true, y_pred)
```

    3/3 [==============================] - 0s 42ms/step
    2/2 [==============================] - 0s 29ms/step
    2/2 [==============================] - 0s 14ms/step
    1/1 [==============================] - 0s 26ms/step
    3/3 [==============================] - 0s 34ms/step
    3/3 [==============================] - 0s 38ms/step
    1/1 [==============================] - 0s 63ms/step
    3/3 [==============================] - 0s 43ms/step
    2/2 [==============================] - 0s 31ms/step
    1/1 [==============================] - 0s 40ms/step
    2/2 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 40ms/step
    2/2 [==============================] - 0s 15ms/step
    2/2 [==============================] - 0s 37ms/step
    2/2 [==============================] - 0s 30ms/step
    3/3 [==============================] - 0s 30ms/step
    1/1 [==============================] - 0s 55ms/step
    1/1 [==============================] - 0s 44ms/step
    1/1 [==============================] - 0s 34ms/step
    3/3 [==============================] - 0s 39ms/step
    2/2 [==============================] - 0s 24ms/step
    2/2 [==============================] - 0s 48ms/step
    1/1 [==============================] - 0s 59ms/step
    1/1 [==============================] - 0s 29ms/step
    1/1 [==============================] - 0s 45ms/step
    2/2 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 56ms/step
    1/1 [==============================] - 0s 59ms/step
    2/2 [==============================] - 0s 41ms/step
    4/4 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 35ms/step
    2/2 [==============================] - 0s 13ms/step
    2/2 [==============================] - 0s 35ms/step
    1/1 [==============================] - 0s 55ms/step
    2/2 [==============================] - 0s 39ms/step
    2/2 [==============================] - 0s 27ms/step
    2/2 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 74ms/step
    2/2 [==============================] - 0s 25ms/step
    4/4 [==============================] - 0s 43ms/step
    2/2 [==============================] - 0s 32ms/step
    2/2 [==============================] - 0s 18ms/step
    3/3 [==============================] - 0s 54ms/step
    1/1 [==============================] - 0s 51ms/step
    2/2 [==============================] - 0s 39ms/step
    2/2 [==============================] - 0s 25ms/step
    2/2 [==============================] - 0s 28ms/step
    3/3 [==============================] - 0s 36ms/step
    2/2 [==============================] - 0s 14ms/step
    1/1 [==============================] - 0s 44ms/step
    2/2 [==============================] - 0s 37ms/step
    1/1 [==============================] - 0s 40ms/step
    2/2 [==============================] - 0s 50ms/step
    2/2 [==============================] - 0s 48ms/step
    2/2 [==============================] - 0s 51ms/step
    2/2 [==============================] - 0s 11ms/step
    2/2 [==============================] - 0s 22ms/step
    2/2 [==============================] - 0s 49ms/step
    2/2 [==============================] - 0s 59ms/step
    1/1 [==============================] - 0s 77ms/step
    1/1 [==============================] - 0s 69ms/step
    2/2 [==============================] - 0s 46ms/step
    2/2 [==============================] - 0s 38ms/step
    2/2 [==============================] - 0s 34ms/step
    1/1 [==============================] - 0s 71ms/step
    2/2 [==============================] - 0s 45ms/step
    2/2 [==============================] - 0s 26ms/step
    2/2 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 60ms/step
    3/3 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 47ms/step
    2/2 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 44ms/step
    2/2 [==============================] - 0s 17ms/step
    3/3 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 47ms/step
    2/2 [==============================] - 0s 33ms/step
    2/2 [==============================] - 0s 28ms/step
    1/1 [==============================] - 0s 66ms/step
    2/2 [==============================] - 0s 24ms/step
    2/2 [==============================] - 0s 32ms/step
    2/2 [==============================] - 0s 19ms/step
    3/3 [==============================] - 0s 34ms/step
    3/3 [==============================] - 0s 34ms/step
    3/3 [==============================] - 0s 31ms/step
    2/2 [==============================] - 0s 30ms/step
    3/3 [==============================] - 0s 35ms/step
    2/2 [==============================] - 0s 24ms/step
    2/2 [==============================] - 0s 46ms/step
    2/2 [==============================] - 0s 33ms/step
    1/1 [==============================] - 0s 41ms/step
    3/3 [==============================] - 0s 44ms/step
    3/3 [==============================] - 0s 33ms/step
    4/4 [==============================] - 0s 36ms/step
    2/2 [==============================] - 0s 28ms/step
    2/2 [==============================] - 0s 8ms/step
    2/2 [==============================] - 0s 30ms/step
    3/3 [==============================] - 0s 32ms/step
    2/2 [==============================] - 0s 26ms/step
    3/3 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 28ms/step
    2/2 [==============================] - 0s 9ms/step
    1/1 [==============================] - 0s 31ms/step
    2/2 [==============================] - 0s 19ms/step
    2/2 [==============================] - 0s 27ms/step
    1/1 [==============================] - 0s 30ms/step
    2/2 [==============================] - 0s 31ms/step
    2/2 [==============================] - 0s 17ms/step
    1/1 [==============================] - 0s 46ms/step
    2/2 [==============================] - 0s 27ms/step
    2/2 [==============================] - 0s 26ms/step
    2/2 [==============================] - 0s 10ms/step
    4/4 [==============================] - 0s 43ms/step
    3/3 [==============================] - 0s 35ms/step
    3/3 [==============================] - 0s 33ms/step
    3/3 [==============================] - 0s 29ms/step
    2/2 [==============================] - 0s 10ms/step
    3/3 [==============================] - 0s 48ms/step
    1/1 [==============================] - 0s 69ms/step
    2/2 [==============================] - 0s 10ms/step
    2/2 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 29ms/step
    4/4 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 25ms/step
    2/2 [==============================] - 0s 11ms/step
    3/3 [==============================] - 0s 54ms/step
    2/2 [==============================] - 0s 23ms/step
    4/4 [==============================] - 0s 35ms/step
    1/1 [==============================] - 0s 53ms/step
    2/2 [==============================] - 0s 13ms/step
    1/1 [==============================] - 0s 41ms/step
    3/3 [==============================] - 0s 34ms/step
    2/2 [==============================] - 0s 53ms/step
    1/1 [==============================] - 0s 65ms/step
    3/3 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 39ms/step
    3/3 [==============================] - 0s 45ms/step
    2/2 [==============================] - 0s 47ms/step
    1/1 [==============================] - 0s 64ms/step
    2/2 [==============================] - 0s 29ms/step
    2/2 [==============================] - 0s 11ms/step
    2/2 [==============================] - 0s 27ms/step
    2/2 [==============================] - 0s 23ms/step
    4/4 [==============================] - 0s 40ms/step
    3/3 [==============================] - 0s 35ms/step
    2/2 [==============================] - 0s 25ms/step
    2/2 [==============================] - 0s 45ms/step
    2/2 [==============================] - 0s 51ms/step
    1/1 [==============================] - 0s 66ms/step
    1/1 [==============================] - 0s 32ms/step
    1/1 [==============================] - 0s 38ms/step
    1/1 [==============================] - 0s 40ms/step
    2/2 [==============================] - 0s 34ms/step
    2/2 [==============================] - 0s 39ms/step
    2/2 [==============================] - 0s 32ms/step
    1/1 [==============================] - 0s 34ms/step
    1/1 [==============================] - 0s 42ms/step
    2/2 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 50ms/step
    1/1 [==============================] - 0s 60ms/step
    3/3 [==============================] - 0s 36ms/step
    2/2 [==============================] - 0s 28ms/step
    1/1 [==============================] - 0s 50ms/step
    1/1 [==============================] - 0s 60ms/step
    1/1 [==============================] - 0s 52ms/step
    2/2 [==============================] - 0s 29ms/step
    1/1 [==============================] - 0s 61ms/step
    1/1 [==============================] - 0s 70ms/step
    3/3 [==============================] - 0s 34ms/step
    1/1 [==============================] - 0s 31ms/step
    2/2 [==============================] - 0s 29ms/step
    3/3 [==============================] - 0s 29ms/step
    3/3 [==============================] - 0s 27ms/step
    2/2 [==============================] - 0s 19ms/step
    2/2 [==============================] - 0s 26ms/step
    2/2 [==============================] - 0s 42ms/step
    3/3 [==============================] - 0s 36ms/step
    1/1 [==============================] - 0s 48ms/step
    2/2 [==============================] - 0s 45ms/step
    3/3 [==============================] - 0s 37ms/step
    1/1 [==============================] - 0s 68ms/step
    2/2 [==============================] - 0s 14ms/step
    2/2 [==============================] - 0s 35ms/step
    1/1 [==============================] - 0s 61ms/step
    2/2 [==============================] - 0s 50ms/step
    1/1 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 23ms/step
    2/2 [==============================] - 0s 46ms/step
    2/2 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 30ms/step
    2/2 [==============================] - 0s 51ms/step
    3/3 [==============================] - 0s 40ms/step
    2/2 [==============================] - 0s 36ms/step
    3/3 [==============================] - 0s 47ms/step
    2/2 [==============================] - 0s 40ms/step
    2/2 [==============================] - 0s 46ms/step
    2/2 [==============================] - 0s 18ms/step
    2/2 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 45ms/step
    1/1 [==============================] - 0s 53ms/step
    2/2 [==============================] - 0s 24ms/step
    2/2 [==============================] - 0s 60ms/step
    2/2 [==============================] - 0s 37ms/step
    2/2 [==============================] - 0s 53ms/step
    1/1 [==============================] - 0s 37ms/step
    3/3 [==============================] - 0s 30ms/step
    1/1 [==============================] - 0s 25ms/step
    3/3 [==============================] - 0s 44ms/step
    3/3 [==============================] - 0s 27ms/step
    1/1 [==============================] - 0s 59ms/step
    3/3 [==============================] - 0s 38ms/step
    3/3 [==============================] - 0s 37ms/step
    2/2 [==============================] - 0s 9ms/step
    3/3 [==============================] - 0s 42ms/step
    2/2 [==============================] - 0s 43ms/step
    2/2 [==============================] - 0s 44ms/step
    2/2 [==============================] - 0s 13ms/step
    2/2 [==============================] - 0s 36ms/step
    2/2 [==============================] - 0s 11ms/step
    2/2 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 57ms/step
    2/2 [==============================] - 0s 24ms/step
    3/3 [==============================] - 0s 43ms/step
    2/2 [==============================] - 0s 17ms/step
    1/1 [==============================] - 0s 59ms/step
    3/3 [==============================] - 0s 40ms/step
    2/2 [==============================] - 0s 29ms/step
    2/2 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 56ms/step
    1/1 [==============================] - 0s 43ms/step
    1/1 [==============================] - 0s 54ms/step
    3/3 [==============================] - 0s 35ms/step
    3/3 [==============================] - 0s 36ms/step
    1/1 [==============================] - 0s 37ms/step
    1/1 [==============================] - 0s 59ms/step
    3/3 [==============================] - 0s 48ms/step
    2/2 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 65ms/step
    1/1 [==============================] - 0s 36ms/step
    1/1 [==============================] - 0s 52ms/step
    2/2 [==============================] - 0s 29ms/step
    1/1 [==============================] - 0s 44ms/step
    1/1 [==============================] - 0s 46ms/step
    2/2 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 38ms/step
    2/2 [==============================] - 0s 16ms/step
    1/1 [==============================] - 0s 44ms/step
    1/1 [==============================] - 0s 49ms/step
    1/1 [==============================] - 0s 57ms/step
    1/1 [==============================] - 0s 45ms/step
    2/2 [==============================] - 0s 30ms/step
    2/2 [==============================] - 0s 17ms/step
    3/3 [==============================] - 0s 51ms/step
    2/2 [==============================] - 0s 55ms/step
    2/2 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 35ms/step
    4/4 [==============================] - 0s 42ms/step
    3/3 [==============================] - 0s 50ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 67ms/step
    2/2 [==============================] - 0s 12ms/step
    1/1 [==============================] - 0s 63ms/step
    3/3 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 51ms/step
    2/2 [==============================] - 0s 30ms/step
    2/2 [==============================] - 0s 45ms/step
    2/2 [==============================] - 0s 35ms/step
    3/3 [==============================] - 0s 42ms/step
    4/4 [==============================] - 0s 36ms/step
    2/2 [==============================] - 0s 30ms/step
    1/1 [==============================] - 0s 36ms/step
    1/1 [==============================] - 0s 50ms/step
    3/3 [==============================] - 0s 48ms/step
    3/3 [==============================] - 0s 45ms/step
    3/3 [==============================] - 0s 39ms/step
    2/2 [==============================] - 0s 21ms/step
    3/3 [==============================] - 0s 41ms/step
    4/4 [==============================] - 0s 44ms/step
    1/1 [==============================] - 0s 40ms/step
    2/2 [==============================] - 0s 7ms/step
    1/1 [==============================] - 0s 37ms/step
    2/2 [==============================] - 0s 7ms/step
    1/1 [==============================] - 0s 66ms/step
    3/3 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 34ms/step
    1/1 [==============================] - 0s 47ms/step
    3/3 [==============================] - 0s 32ms/step
    1/1 [==============================] - 0s 56ms/step
    2/2 [==============================] - 0s 24ms/step
    2/2 [==============================] - 0s 33ms/step
    2/2 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 47ms/step
    2/2 [==============================] - 0s 53ms/step
    2/2 [==============================] - 0s 12ms/step
    1/1 [==============================] - 0s 63ms/step
    1/1 [==============================] - 0s 54ms/step
    2/2 [==============================] - 0s 28ms/step
    1/1 [==============================] - 0s 47ms/step
    1/1 [==============================] - 0s 64ms/step
    2/2 [==============================] - 0s 7ms/step
    1/1 [==============================] - 0s 31ms/step
    4/4 [==============================] - 0s 46ms/step
    2/2 [==============================] - 0s 20ms/step
    2/2 [==============================] - 0s 51ms/step
    1/1 [==============================] - 0s 56ms/step
    1/1 [==============================] - 0s 38ms/step
    2/2 [==============================] - 0s 61ms/step
    3/3 [==============================] - 0s 38ms/step
    2/2 [==============================] - 0s 50ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 59ms/step
    3/3 [==============================] - 0s 29ms/step
    2/2 [==============================] - 0s 20ms/step
    2/2 [==============================] - 0s 18ms/step
    3/3 [==============================] - 0s 38ms/step
    2/2 [==============================] - 0s 11ms/step
    2/2 [==============================] - 0s 25ms/step
    3/3 [==============================] - 0s 46ms/step
    4/4 [==============================] - 0s 38ms/step
    1/1 [==============================] - 0s 35ms/step
    2/2 [==============================] - 0s 48ms/step
    2/2 [==============================] - 0s 52ms/step
    2/2 [==============================] - 0s 14ms/step
    2/2 [==============================] - 0s 41ms/step
    1/1 [==============================] - 0s 48ms/step
    2/2 [==============================] - 0s 52ms/step
    2/2 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 31ms/step
    2/2 [==============================] - 0s 41ms/step
    1/1 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 32ms/step
    1/1 [==============================] - 0s 45ms/step
    2/2 [==============================] - 0s 48ms/step
    2/2 [==============================] - 0s 32ms/step
    1/1 [==============================] - 0s 49ms/step
    1/1 [==============================] - 0s 51ms/step
    2/2 [==============================] - 0s 50ms/step
    2/2 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 35ms/step
    2/2 [==============================] - 0s 30ms/step
    1/1 [==============================] - 0s 36ms/step
    2/2 [==============================] - 0s 7ms/step
    1/1 [==============================] - 0s 45ms/step
    2/2 [==============================] - 0s 46ms/step
    1/1 [==============================] - 0s 47ms/step
    2/2 [==============================] - 0s 27ms/step
    1/1 [==============================] - 0s 37ms/step
    3/3 [==============================] - 0s 34ms/step
    2/2 [==============================] - 0s 10ms/step
    3/3 [==============================] - 0s 46ms/step
    3/3 [==============================] - 0s 42ms/step
    2/2 [==============================] - 0s 45ms/step
    1/1 [==============================] - 0s 68ms/step
    3/3 [==============================] - 0s 34ms/step
    1/1 [==============================] - 0s 50ms/step
    1/1 [==============================] - 0s 51ms/step
    1/1 [==============================] - 0s 59ms/step
    2/2 [==============================] - 0s 9ms/step
    2/2 [==============================] - 0s 29ms/step
    4/4 [==============================] - 0s 53ms/step
    1/1 [==============================] - 0s 66ms/step
    3/3 [==============================] - 0s 43ms/step
    2/2 [==============================] - 0s 36ms/step
    1/1 [==============================] - 0s 58ms/step
    2/2 [==============================] - 0s 38ms/step
    1/1 [==============================] - 0s 75ms/step
    2/2 [==============================] - 0s 43ms/step
    2/2 [==============================] - 0s 54ms/step
    2/2 [==============================] - 0s 29ms/step
    2/2 [==============================] - 0s 30ms/step
    3/3 [==============================] - 0s 51ms/step
    2/2 [==============================] - 0s 57ms/step
    2/2 [==============================] - 0s 7ms/step
    1/1 [==============================] - 0s 30ms/step
    2/2 [==============================] - 0s 32ms/step
    1/1 [==============================] - 0s 68ms/step
    2/2 [==============================] - 0s 43ms/step
    1/1 [==============================] - 0s 69ms/step
    4/4 [==============================] - 0s 47ms/step
    3/3 [==============================] - 0s 28ms/step
    2/2 [==============================] - 0s 37ms/step
    2/2 [==============================] - 0s 39ms/step
    2/2 [==============================] - 0s 22ms/step
    2/2 [==============================] - 0s 6ms/step
    2/2 [==============================] - 0s 25ms/step
    1/1 [==============================] - 0s 70ms/step
    2/2 [==============================] - 0s 20ms/step
    2/2 [==============================] - 0s 24ms/step
    2/2 [==============================] - 0s 47ms/step
    2/2 [==============================] - 0s 44ms/step
    2/2 [==============================] - 0s 30ms/step
    1/1 [==============================] - 0s 48ms/step
    3/3 [==============================] - 0s 44ms/step
    1/1 [==============================] - 0s 57ms/step
    2/2 [==============================] - 0s 26ms/step
    2/2 [==============================] - 0s 53ms/step
    3/3 [==============================] - 0s 38ms/step
    1/1 [==============================] - 0s 66ms/step
    2/2 [==============================] - 0s 34ms/step
    2/2 [==============================] - 0s 34ms/step
    2/2 [==============================] - 0s 25ms/step
    2/2 [==============================] - 0s 23ms/step
    2/2 [==============================] - 0s 23ms/step
    2/2 [==============================] - 0s 44ms/step
    3/3 [==============================] - 0s 36ms/step
    3/3 [==============================] - 0s 36ms/step
    2/2 [==============================] - 0s 26ms/step
    2/2 [==============================] - 0s 33ms/step
    2/2 [==============================] - 0s 22ms/step
    4/4 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 14ms/step
    1/1 [==============================] - 0s 36ms/step
    1/1 [==============================] - 0s 59ms/step
    2/2 [==============================] - 0s 40ms/step
    2/2 [==============================] - 0s 27ms/step
    2/2 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 33ms/step
    2/2 [==============================] - 0s 24ms/step
    3/3 [==============================] - 0s 31ms/step
    4/4 [==============================] - 0s 48ms/step
    1/1 [==============================] - 0s 63ms/step
    2/2 [==============================] - 0s 11ms/step
    4/4 [==============================] - 0s 48ms/step
    1/1 [==============================] - 0s 51ms/step
    2/2 [==============================] - 0s 16ms/step
    1/1 [==============================] - 0s 73ms/step
    3/3 [==============================] - 0s 33ms/step
    1/1 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 60ms/step
    2/2 [==============================] - 0s 44ms/step
    2/2 [==============================] - 0s 31ms/step
    2/2 [==============================] - 0s 18ms/step
    2/2 [==============================] - 0s 37ms/step
    2/2 [==============================] - 0s 31ms/step
    2/2 [==============================] - 0s 49ms/step
    2/2 [==============================] - 0s 22ms/step
    3/3 [==============================] - 0s 29ms/step
    2/2 [==============================] - 0s 19ms/step
    2/2 [==============================] - 0s 18ms/step
    3/3 [==============================] - 0s 47ms/step
    3/3 [==============================] - 0s 49ms/step
    2/2 [==============================] - 0s 9ms/step
    1/1 [==============================] - 0s 62ms/step
    2/2 [==============================] - 0s 30ms/step
    3/3 [==============================] - 0s 48ms/step
    3/3 [==============================] - 0s 45ms/step
    3/3 [==============================] - 0s 41ms/step
    1/1 [==============================] - 0s 57ms/step
    1/1 [==============================] - 0s 37ms/step
    2/2 [==============================] - 0s 40ms/step
    2/2 [==============================] - 0s 33ms/step
    2/2 [==============================] - 0s 34ms/step
    2/2 [==============================] - 0s 27ms/step
    3/3 [==============================] - 0s 35ms/step
    2/2 [==============================] - 0s 29ms/step
    2/2 [==============================] - 0s 34ms/step
    3/3 [==============================] - 0s 51ms/step
    3/3 [==============================] - 0s 42ms/step
    2/2 [==============================] - 0s 24ms/step
    3/3 [==============================] - 0s 38ms/step
    1/1 [==============================] - 0s 25ms/step
    2/2 [==============================] - 0s 40ms/step
    2/2 [==============================] - 0s 24ms/step
    2/2 [==============================] - 0s 28ms/step
    3/3 [==============================] - 0s 36ms/step
    2/2 [==============================] - 0s 14ms/step
    1/1 [==============================] - 0s 51ms/step
    2/2 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 52ms/step
    3/3 [==============================] - 0s 36ms/step
    2/2 [==============================] - 0s 59ms/step
    2/2 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 35ms/step
    2/2 [==============================] - 0s 58ms/step
    3/3 [==============================] - 0s 41ms/step
    1/1 [==============================] - 0s 35ms/step
    1/1 [==============================] - 0s 39ms/step
    2/2 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 61ms/step
    2/2 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 52ms/step
    2/2 [==============================] - 0s 43ms/step
    1/1 [==============================] - 0s 27ms/step
    2/2 [==============================] - 0s 35ms/step
    2/2 [==============================] - 0s 7ms/step
    1/1 [==============================] - 0s 74ms/step
    3/3 [==============================] - 0s 39ms/step
    2/2 [==============================] - 0s 42ms/step
    2/2 [==============================] - 0s 55ms/step
    2/2 [==============================] - 0s 20ms/step
    2/2 [==============================] - 0s 36ms/step
    2/2 [==============================] - 0s 38ms/step
    3/3 [==============================] - 0s 51ms/step
    2/2 [==============================] - 0s 32ms/step
    2/2 [==============================] - 0s 19ms/step
    3/3 [==============================] - 0s 40ms/step
    3/3 [==============================] - 0s 40ms/step
    2/2 [==============================] - 0s 33ms/step
    2/2 [==============================] - 0s 24ms/step
    2/2 [==============================] - 0s 6ms/step
    4/4 [==============================] - 0s 38ms/step
    2/2 [==============================] - 0s 36ms/step
    1/1 [==============================] - 0s 65ms/step
    2/2 [==============================] - 0s 23ms/step
    2/2 [==============================] - 0s 31ms/step
    1/1 [==============================] - 0s 24ms/step
    2/2 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 79ms/step
    1/1 [==============================] - 0s 59ms/step
    3/3 [==============================] - 0s 32ms/step
    2/2 [==============================] - 0s 47ms/step
    2/2 [==============================] - 0s 31ms/step
    1/1 [==============================] - 0s 59ms/step
    3/3 [==============================] - 0s 47ms/step
    2/2 [==============================] - 0s 46ms/step
    2/2 [==============================] - 0s 56ms/step
    2/2 [==============================] - 0s 44ms/step
    1/1 [==============================] - 0s 36ms/step
    2/2 [==============================] - 0s 48ms/step
    2/2 [==============================] - 0s 41ms/step
    1/1 [==============================] - 0s 39ms/step
    2/2 [==============================] - 0s 32ms/step
    2/2 [==============================] - 0s 57ms/step
    2/2 [==============================] - 0s 35ms/step
    2/2 [==============================] - 0s 50ms/step
    1/1 [==============================] - 0s 60ms/step
    1/1 [==============================] - 0s 52ms/step
    3/3 [==============================] - 0s 35ms/step
    3/3 [==============================] - 0s 39ms/step
    2/2 [==============================] - 0s 42ms/step
    2/2 [==============================] - 0s 30ms/step
    4/4 [==============================] - 0s 40ms/step
    2/2 [==============================] - 0s 21ms/step
    2/2 [==============================] - 0s 28ms/step
    2/2 [==============================] - 0s 62ms/step
    2/2 [==============================] - 0s 46ms/step
    2/2 [==============================] - 0s 55ms/step
    2/2 [==============================] - 0s 39ms/step
    2/2 [==============================] - 0s 12ms/step
    1/1 [==============================] - 0s 48ms/step
    1/1 [==============================] - 0s 63ms/step
    2/2 [==============================] - 0s 30ms/step
    2/2 [==============================] - 0s 31ms/step
    1/1 [==============================] - 0s 42ms/step
    2/2 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 45ms/step
    2/2 [==============================] - 0s 35ms/step
    4/4 [==============================] - 0s 39ms/step
    2/2 [==============================] - 0s 21ms/step
    2/2 [==============================] - 0s 51ms/step
    2/2 [==============================] - 0s 20ms/step
    3/3 [==============================] - 0s 49ms/step
    3/3 [==============================] - 0s 33ms/step
    2/2 [==============================] - 0s 25ms/step
    2/2 [==============================] - 0s 48ms/step
    2/2 [==============================] - 0s 26ms/step
    1/1 [==============================] - 0s 49ms/step
    1/1 [==============================] - 0s 61ms/step
    2/2 [==============================] - 0s 53ms/step
    2/2 [==============================] - 0s 48ms/step
    1/1 [==============================] - 0s 35ms/step
    3/3 [==============================] - 0s 48ms/step
    1/1 [==============================] - 0s 34ms/step
    2/2 [==============================] - 0s 37ms/step
    2/2 [==============================] - 0s 23ms/step
    2/2 [==============================] - 0s 16ms/step
    2/2 [==============================] - 0s 38ms/step
    1/1 [==============================] - 0s 45ms/step
    2/2 [==============================] - 0s 20ms/step
    2/2 [==============================] - 0s 22ms/step
    2/2 [==============================] - 0s 35ms/step
    2/2 [==============================] - 0s 33ms/step
    1/1 [==============================] - 0s 48ms/step
    1/1 [==============================] - 0s 63ms/step
    2/2 [==============================] - 0s 23ms/step
    2/2 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 32ms/step
    2/2 [==============================] - 0s 28ms/step
    2/2 [==============================] - 0s 9ms/step
    1/1 [==============================] - 0s 49ms/step
    2/2 [==============================] - 0s 10ms/step
    3/3 [==============================] - 0s 32ms/step
    2/2 [==============================] - 0s 15ms/step
    1/1 [==============================] - 0s 34ms/step
    2/2 [==============================] - 0s 9ms/step
    1/1 [==============================] - 0s 55ms/step
    2/2 [==============================] - 0s 33ms/step
    2/2 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 8ms/step
    2/2 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 38ms/step
    2/2 [==============================] - 0s 25ms/step
    2/2 [==============================] - 0s 27ms/step
    2/2 [==============================] - 0s 14ms/step
    4/4 [==============================] - 0s 35ms/step
    2/2 [==============================] - 0s 19ms/step
    2/2 [==============================] - 0s 14ms/step
    3/3 [==============================] - 0s 33ms/step
    3/3 [==============================] - 0s 39ms/step
    3/3 [==============================] - 0s 30ms/step
    3/3 [==============================] - 0s 52ms/step
    3/3 [==============================] - 0s 35ms/step
    2/2 [==============================] - 0s 32ms/step
    2/2 [==============================] - 0s 35ms/step
    1/1 [==============================] - 0s 33ms/step
    1/1 [==============================] - 0s 45ms/step
    1/1 [==============================] - 0s 33ms/step
    1/1 [==============================] - 0s 51ms/step
    2/2 [==============================] - 0s 25ms/step
    2/2 [==============================] - 0s 40ms/step
    3/3 [==============================] - 0s 32ms/step
    3/3 [==============================] - 0s 36ms/step
    2/2 [==============================] - 0s 33ms/step
    1/1 [==============================] - 0s 53ms/step
    1/1 [==============================] - 0s 23ms/step
    3/3 [==============================] - 0s 32ms/step
    2/2 [==============================] - 0s 53ms/step
    2/2 [==============================] - 0s 16ms/step
    2/2 [==============================] - 0s 52ms/step
    2/2 [==============================] - 0s 52ms/step
    2/2 [==============================] - 0s 11ms/step
    1/1 [==============================] - 0s 61ms/step
    2/2 [==============================] - 0s 33ms/step
    2/2 [==============================] - 0s 17ms/step
    3/3 [==============================] - 0s 42ms/step
    2/2 [==============================] - 0s 38ms/step
    1/1 [==============================] - 0s 37ms/step
    1/1 [==============================] - 0s 50ms/step
    2/2 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 49ms/step
    2/2 [==============================] - 0s 13ms/step
    3/3 [==============================] - 0s 32ms/step
    1/1 [==============================] - 0s 38ms/step
    2/2 [==============================] - 0s 17ms/step
    1/1 [==============================] - 0s 30ms/step
    1/1 [==============================] - 0s 29ms/step
    1/1 [==============================] - 0s 60ms/step
    2/2 [==============================] - 0s 30ms/step
    2/2 [==============================] - 0s 45ms/step
    2/2 [==============================] - 0s 28ms/step
    2/2 [==============================] - 0s 30ms/step
    2/2 [==============================] - 0s 18ms/step
    2/2 [==============================] - 0s 34ms/step
    3/3 [==============================] - 0s 42ms/step
    2/2 [==============================] - 0s 13ms/step
    2/2 [==============================] - 0s 33ms/step
    2/2 [==============================] - 0s 12ms/step
    3/3 [==============================] - 0s 43ms/step
    1/1 [==============================] - 0s 43ms/step
    2/2 [==============================] - 0s 29ms/step
    4/4 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 44ms/step
    2/2 [==============================] - 0s 45ms/step
    2/2 [==============================] - 0s 32ms/step
    2/2 [==============================] - 0s 28ms/step
    2/2 [==============================] - 0s 13ms/step
    1/1 [==============================] - 0s 68ms/step
    2/2 [==============================] - 0s 30ms/step
    3/3 [==============================] - 0s 29ms/step
    2/2 [==============================] - 0s 26ms/step
    2/2 [==============================] - 0s 36ms/step
    1/1 [==============================] - 0s 63ms/step
    1/1 [==============================] - 0s 28ms/step
    1/1 [==============================] - 0s 61ms/step
    3/3 [==============================] - 0s 54ms/step
    3/3 [==============================] - 0s 34ms/step
    3/3 [==============================] - 0s 51ms/step
    1/1 [==============================] - 0s 66ms/step
    2/2 [==============================] - 0s 17ms/step
    3/3 [==============================] - 0s 33ms/step
    1/1 [==============================] - 0s 65ms/step
    2/2 [==============================] - 0s 49ms/step
    2/2 [==============================] - 0s 11ms/step
    1/1 [==============================] - 0s 55ms/step
    1/1 [==============================] - 0s 43ms/step
    3/3 [==============================] - 0s 47ms/step
    1/1 [==============================] - 0s 66ms/step
    1/1 [==============================] - 0s 45ms/step
    2/2 [==============================] - 0s 16ms/step
    1/1 [==============================] - 0s 65ms/step
    2/2 [==============================] - 0s 44ms/step
    2/2 [==============================] - 0s 44ms/step
    2/2 [==============================] - 0s 48ms/step
    1/1 [==============================] - 0s 36ms/step
    2/2 [==============================] - 0s 14ms/step
    3/3 [==============================] - 0s 43ms/step
    3/3 [==============================] - 0s 51ms/step
    2/2 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 59ms/step
    2/2 [==============================] - 0s 47ms/step
    2/2 [==============================] - 0s 14ms/step
    1/1 [==============================] - 0s 50ms/step
    2/2 [==============================] - 0s 11ms/step
    1/1 [==============================] - 0s 34ms/step
    2/2 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 66ms/step
    2/2 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 67ms/step
    2/2 [==============================] - 0s 45ms/step
    2/2 [==============================] - 0s 29ms/step
    2/2 [==============================] - 0s 54ms/step
    2/2 [==============================] - 0s 44ms/step
    2/2 [==============================] - 0s 43ms/step
    3/3 [==============================] - 0s 29ms/step
    2/2 [==============================] - 0s 36ms/step
    3/3 [==============================] - 0s 31ms/step
    3/3 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 44ms/step
    2/2 [==============================] - 0s 35ms/step
    4/4 [==============================] - 0s 38ms/step
    2/2 [==============================] - 0s 46ms/step
    1/1 [==============================] - 0s 31ms/step
    2/2 [==============================] - 0s 21ms/step
    3/3 [==============================] - 0s 31ms/step
    2/2 [==============================] - 0s 38ms/step
    3/3 [==============================] - 0s 47ms/step
    2/2 [==============================] - 0s 35ms/step
    2/2 [==============================] - 0s 44ms/step
    1/1 [==============================] - 0s 49ms/step
    2/2 [==============================] - 0s 46ms/step
    2/2 [==============================] - 0s 26ms/step
    2/2 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 70ms/step
    2/2 [==============================] - 0s 41ms/step
    1/1 [==============================] - 0s 48ms/step
    2/2 [==============================] - 0s 33ms/step
    4/4 [==============================] - 0s 36ms/step
    3/3 [==============================] - 0s 38ms/step
    2/2 [==============================] - 0s 17ms/step
    3/3 [==============================] - 0s 33ms/step
    2/2 [==============================] - 0s 30ms/step
    3/3 [==============================] - 0s 38ms/step
    2/2 [==============================] - 0s 24ms/step
    2/2 [==============================] - 0s 34ms/step
    4/4 [==============================] - 0s 45ms/step
    2/2 [==============================] - 0s 36ms/step
    2/2 [==============================] - 0s 46ms/step
    2/2 [==============================] - 0s 9ms/step
    1/1 [==============================] - 0s 62ms/step
    2/2 [==============================] - 0s 44ms/step
    1/1 [==============================] - 0s 55ms/step
    2/2 [==============================] - 0s 26ms/step
    2/2 [==============================] - 0s 47ms/step
    2/2 [==============================] - 0s 26ms/step
    1/1 [==============================] - 0s 55ms/step
    1/1 [==============================] - 0s 50ms/step
    2/2 [==============================] - 0s 14ms/step
    2/2 [==============================] - 0s 38ms/step
    2/2 [==============================] - 0s 33ms/step
    2/2 [==============================] - 0s 20ms/step
    2/2 [==============================] - 0s 27ms/step
    2/2 [==============================] - 0s 20ms/step
    2/2 [==============================] - 0s 46ms/step
    2/2 [==============================] - 0s 13ms/step
    2/2 [==============================] - 0s 28ms/step
    2/2 [==============================] - 0s 20ms/step
    3/3 [==============================] - 0s 33ms/step
    2/2 [==============================] - 0s 18ms/step
    2/2 [==============================] - 0s 46ms/step
    3/3 [==============================] - 0s 40ms/step
    2/2 [==============================] - 0s 32ms/step
    2/2 [==============================] - 0s 27ms/step
    1/1 [==============================] - 0s 56ms/step
    1/1 [==============================] - 0s 35ms/step
    1/1 [==============================] - 0s 29ms/step
    4/4 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 46ms/step
    2/2 [==============================] - 0s 15ms/step
    4/4 [==============================] - 0s 40ms/step
    2/2 [==============================] - 0s 49ms/step
    3/3 [==============================] - 0s 48ms/step
    1/1 [==============================] - 0s 48ms/step
    2/2 [==============================] - 0s 30ms/step
    1/1 [==============================] - 0s 61ms/step
    2/2 [==============================] - 0s 26ms/step
    1/1 [==============================] - 0s 23ms/step
    2/2 [==============================] - 0s 27ms/step
    3/3 [==============================] - 0s 43ms/step
    2/2 [==============================] - 0s 12ms/step
    3/3 [==============================] - 0s 47ms/step
    2/2 [==============================] - 0s 26ms/step
    2/2 [==============================] - 0s 24ms/step
    3/3 [==============================] - 0s 45ms/step
    1/1 [==============================] - 0s 64ms/step
    2/2 [==============================] - 0s 31ms/step
    3/3 [==============================] - 0s 42ms/step
    2/2 [==============================] - 0s 23ms/step
    3/3 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 47ms/step
    2/2 [==============================] - 0s 33ms/step
    2/2 [==============================] - 0s 27ms/step
    3/3 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 55ms/step
    2/2 [==============================] - 0s 55ms/step
    2/2 [==============================] - 0s 35ms/step
    1/1 [==============================] - 0s 53ms/step
    2/2 [==============================] - 0s 35ms/step
    2/2 [==============================] - 0s 22ms/step
    2/2 [==============================] - 0s 27ms/step
    3/3 [==============================] - 0s 51ms/step
    2/2 [==============================] - 0s 50ms/step
    2/2 [==============================] - 0s 27ms/step
    2/2 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 46ms/step
    1/1 [==============================] - 0s 60ms/step
    1/1 [==============================] - 0s 32ms/step
    1/1 [==============================] - 0s 58ms/step
    2/2 [==============================] - 0s 33ms/step
    3/3 [==============================] - 0s 50ms/step
    2/2 [==============================] - 0s 55ms/step
    2/2 [==============================] - 0s 27ms/step
    1/1 [==============================] - 0s 34ms/step
    4/4 [==============================] - 0s 38ms/step
    2/2 [==============================] - 0s 32ms/step
    1/1 [==============================] - 0s 42ms/step
    2/2 [==============================] - 0s 35ms/step
    2/2 [==============================] - 0s 22ms/step
    2/2 [==============================] - 0s 61ms/step
    2/2 [==============================] - 0s 42ms/step
    2/2 [==============================] - 0s 27ms/step
    2/2 [==============================] - 0s 43ms/step
    2/2 [==============================] - 0s 19ms/step
    3/3 [==============================] - 0s 41ms/step
    1/1 [==============================] - 0s 67ms/step
    2/2 [==============================] - 0s 49ms/step
    2/2 [==============================] - 0s 26ms/step
    3/3 [==============================] - 0s 33ms/step
    1/1 [==============================] - 0s 50ms/step
    3/3 [==============================] - 0s 40ms/step
    2/2 [==============================] - 0s 36ms/step
    3/3 [==============================] - 0s 56ms/step
    3/3 [==============================] - 0s 53ms/step
    2/2 [==============================] - 0s 20ms/step
    2/2 [==============================] - 0s 54ms/step
    4/4 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 34ms/step
    4/4 [==============================] - 0s 42ms/step
    4/4 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 35ms/step
    3/3 [==============================] - 0s 31ms/step
    2/2 [==============================] - 0s 35ms/step
    2/2 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 73ms/step
    1/1 [==============================] - 0s 71ms/step
    2/2 [==============================] - 0s 21ms/step
    3/3 [==============================] - 0s 33ms/step
    2/2 [==============================] - 0s 37ms/step
    1/1 [==============================] - 0s 73ms/step
    2/2 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 33ms/step
    2/2 [==============================] - 0s 34ms/step
    2/2 [==============================] - 0s 37ms/step
    2/2 [==============================] - 0s 43ms/step
    2/2 [==============================] - 0s 9ms/step
    3/3 [==============================] - 0s 37ms/step
    1/1 [==============================] - 0s 78ms/step
    4/4 [==============================] - 0s 55ms/step
    1/1 [==============================] - 0s 60ms/step
    2/2 [==============================] - 0s 31ms/step
    1/1 [==============================] - 0s 67ms/step
    2/2 [==============================] - 0s 18ms/step
    2/2 [==============================] - 0s 23ms/step
    2/2 [==============================] - 0s 15ms/step
    1/1 [==============================] - 0s 66ms/step
    3/3 [==============================] - 0s 46ms/step
    2/2 [==============================] - 0s 46ms/step
    2/2 [==============================] - 0s 25ms/step
    2/2 [==============================] - 0s 17ms/step
    4/4 [==============================] - 0s 42ms/step
    2/2 [==============================] - 0s 11ms/step
    2/2 [==============================] - 0s 38ms/step
    2/2 [==============================] - 0s 25ms/step
    2/2 [==============================] - 0s 46ms/step
    2/2 [==============================] - 0s 37ms/step
    2/2 [==============================] - 0s 47ms/step
    1/1 [==============================] - 0s 75ms/step
    2/2 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 52ms/step
    1/1 [==============================] - 0s 55ms/step
    3/3 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 63ms/step
    1/1 [==============================] - 0s 44ms/step
    1/1 [==============================] - 0s 51ms/step
    1/1 [==============================] - 0s 37ms/step
    2/2 [==============================] - 0s 61ms/step
    2/2 [==============================] - 0s 56ms/step
    1/1 [==============================] - 0s 70ms/step
    1/1 [==============================] - 0s 49ms/step
    3/3 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 65ms/step
    1/1 [==============================] - 0s 52ms/step
    2/2 [==============================] - 0s 58ms/step
    1/1 [==============================] - 0s 60ms/step
    1/1 [==============================] - 0s 71ms/step
    3/3 [==============================] - 0s 38ms/step
    1/1 [==============================] - 0s 50ms/step
    2/2 [==============================] - 0s 35ms/step
    3/3 [==============================] - 0s 35ms/step
    3/3 [==============================] - 0s 34ms/step
    2/2 [==============================] - 0s 10ms/step
    2/2 [==============================] - 0s 27ms/step
    2/2 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 53ms/step
    2/2 [==============================] - 0s 24ms/step
    2/2 [==============================] - 0s 8ms/step
    3/3 [==============================] - 0s 30ms/step
    3/3 [==============================] - 0s 43ms/step
    2/2 [==============================] - 0s 36ms/step
    1/1 [==============================] - 0s 65ms/step
    3/3 [==============================] - 0s 47ms/step
    2/2 [==============================] - 0s 57ms/step
    1/1 [==============================] - 0s 62ms/step
    2/2 [==============================] - 0s 31ms/step
    1/1 [==============================] - 0s 59ms/step
    1/1 [==============================] - 0s 62ms/step
    3/3 [==============================] - 0s 33ms/step
    2/2 [==============================] - 0s 17ms/step
    2/2 [==============================] - 0s 10ms/step
    1/1 [==============================] - 0s 51ms/step
    3/3 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 41ms/step
    4/4 [==============================] - 0s 42ms/step
    2/2 [==============================] - 0s 44ms/step
    2/2 [==============================] - 0s 49ms/step
    2/2 [==============================] - 0s 35ms/step
    1/1 [==============================] - 0s 38ms/step
    2/2 [==============================] - 0s 23ms/step
    2/2 [==============================] - 0s 22ms/step
    2/2 [==============================] - 0s 50ms/step
    2/2 [==============================] - 0s 60ms/step
    2/2 [==============================] - 0s 40ms/step
    2/2 [==============================] - 0s 24ms/step
    2/2 [==============================] - 0s 34ms/step
    2/2 [==============================] - 0s 26ms/step
    1/1 [==============================] - 0s 61ms/step
    2/2 [==============================] - 0s 14ms/step
    2/2 [==============================] - 0s 22ms/step
    2/2 [==============================] - 0s 33ms/step
    3/3 [==============================] - 0s 43ms/step
    3/3 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 53ms/step
    2/2 [==============================] - 0s 35ms/step
    3/3 [==============================] - 0s 30ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 23ms/step
    2/2 [==============================] - 0s 44ms/step
    1/1 [==============================] - 0s 36ms/step
    2/2 [==============================] - 0s 22ms/step
    2/2 [==============================] - 0s 46ms/step
    1/1 [==============================] - 0s 37ms/step
    2/2 [==============================] - 0s 14ms/step
    2/2 [==============================] - 0s 62ms/step
    2/2 [==============================] - 0s 17ms/step
    2/2 [==============================] - 0s 37ms/step
    1/1 [==============================] - 0s 71ms/step
    2/2 [==============================] - 0s 36ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 66ms/step
    1/1 [==============================] - 0s 30ms/step
    3/3 [==============================] - 0s 35ms/step
    2/2 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 31ms/step
    5/5 [==============================] - 0s 41ms/step
    2/2 [==============================] - 0s 12ms/step
    1/1 [==============================] - 0s 37ms/step
    1/1 [==============================] - 0s 48ms/step
    3/3 [==============================] - 0s 35ms/step
    1/1 [==============================] - 0s 51ms/step
    3/3 [==============================] - 0s 46ms/step
    1/1 [==============================] - 0s 59ms/step
    1/1 [==============================] - 0s 42ms/step
    3/3 [==============================] - 0s 31ms/step
    1/1 [==============================] - 0s 46ms/step
    3/3 [==============================] - 0s 33ms/step
    2/2 [==============================] - 0s 21ms/step
    




    0.7390417940876657




```python
from keras.models import load_model
model = load_model("my_model.h5")
```


```python
model = halfDeepWriter((113, 113, 1), 25)  # Assuming you changed the output size to 25
model.load_weights("/content/best.hdf5")
```


```python
unique_writer_ids = np.unique(y)
writer_mapping = {index: writer_id for index, writer_id in enumerate(unique_writer_ids)}
```


```python
from keras.preprocessing import image
import matplotlib.pyplot as plt

#img_path = './data/Demo/a01-102u-s01-00.png'
#img_path = './data/Demo/a01-003u-s00-00.png'
img_path = './data/Demo/a01-043u-s00-00.png'
img = image.load_img(img_path, target_size=(113, 113), color_mode="grayscale")
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Normalize to [0, 1]
img_array = np.expand_dims(img_array, axis=0)  # Model expects a batch of images
plt.imshow(img, cmap='gray')  # 'cmap' is set to 'gray' since your images are grayscale
plt.axis('off')  # To turn off axis numbering
plt.show()
```


    
![png](output_15_0.png)
    



```python
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
```

    1/1 [==============================] - 0s 20ms/step
    


```python
predicted_writer_id = writer_mapping[predicted_class]
print(f"The predicted writer is: {predicted_writer_id}")
```

    The predicted writer is: temp_sentences\m06
    


```python

```
