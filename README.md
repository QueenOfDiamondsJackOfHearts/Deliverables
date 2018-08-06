# Deliverables

## Contents 
* Notebook on Training and Evaluating a simple convolutional model with keras
* Notebook on building a transfer learning model using a keras pre-trained model
* Notebook on Data Gathering

## How to use 

### Step 1. Download Dependencies

To run these notebooks you are required to have:
* anaconda with jupyternotebook
* Tensorflow
* Keras

After installing anaconda (https://conda.io/docs/user-guide/install/index.html), it is most simple to type "pip install <package>" into the terminal. This will download the dependecies for these packages globally. One could also install these packages to a specific environment using conda. 
  
### Step 2. Download Models and Data Set from google drive

The Url is https://drive.google.com/open?id=1yzxTuUINb9uF1EDmf9sQAvKXOawtsVW2
The directory "model_LFS" contains the models we have trained and "Data_Set1_LFS" contains our data set divide in a way which makes it easy for keras to handle. These should be placed in the same directory from which you are running the notebooks

### Step 3. Use Jupyter notebook to explore

You should always run the first block, which imports the packages needed. You should also run any function definitions you see. Other than that, the general rule of thumb is that to run a given cell, you must run the one above. This is not always the case as some tasks, like loading a pre-trained model from "model_LFS" has no real requirements except importing the necessary function. 

## Notes:

These models are prediction ready, but the transfer learning model does not quite work with the keras training mode. There seems to be an issue with the inference of the shape of some outputs, so some other method might have to be added to class 
LinearSVM in utils_TFlearn. 
