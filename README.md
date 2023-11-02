# DeepWriter Writing Identifier with Autoencoder
Implementation of ["DeepWriter: A Multi-Stream Deep CNN for Text-independent Writer Identification"](https://arxiv.org/abs/1606.06472)

Contains the DeepWriter and HalfDeepWriter network architecture implementation.

A random image stripping is also implemented.

Need to install
Download [anaconda](https://www.anaconda.com/download)
Install the following commands in terminal
- pip install pillow
- pip install keras
- pip install tensorboard
- pip install tensorflow
- pip install opencv-python

Open anaconda navigator and install keras, tensorboard, tensorflow and opencv-python under environments in base(root)

Next, open visual studio code
Download and open this folder in VSCode
In terminal run ```jupyter notebook```
A notebook should open in your browser and navigate to the DeepWriter.ipynb

Once open follow the instructions in the notebook to download the temp_sentences.zip and Demo.zip

Press the Kernel drop down then finally, "Restart & Run All"

Follow the rest of the instructions in the notebook to run a demo

<!--
NOTES
Modify the loadData() function to work with dataset.
The X must contain image of shape (row, column, 1) and y must contain target class.
-->
