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
* google-api-python-client
* matplotlib
* sklearn

After installing anaconda (https://conda.io/docs/user-guide/install/index.html), it is most simple to type "pip install <package>" into the terminal. This will download the dependecies for these packages globally. One could also install these packages to a specific environment using conda. 
  
### Step 2. Download Models and Data Set from google drive

The Url is https://drive.google.com/open?id=1yzxTuUINb9uF1EDmf9sQAvKXOawtsVW2
The directory "model_LFS" contains the models we have trained and "Data_Set1" contains our data set divide in a way which makes it easy for keras to handle. These should be placed in the same directory from which you are running the notebooks. 

### Step 3. Use Jupyter notebook to explore the models

You should always run the first block, which imports the packages needed. You should also run any function definitions you see. Other than that, the general rule of thumb is that to run a given cell, you must run the one above. This is not always the case as some tasks, like loading a pre-trained model from "model_LFS" has no real requirements except importing the necessary modules. 

### Step 4. Exploring the Data Gathering folder

To use this notebook, you must set your own Custom Search Engine. A tutorial can be found: here https://developers.google.com/custom-search/docs/tutorial/creatingcse. This will involve both aquiring an API key, and creating an actual search engine. There is a relatively simple console that can be used to make search engines quickly, with the ability to create an XML file to get a particularly fine tuned search engine. In the current work, I have not explored this functionality in depth, but it may be interesting. Check the Utility module (GCS_utils) for details on the functions used to create the dataset, as well as the functions used for file manipulations.

## Notes:

These models are prediction ready, but the transfer learning model does not quite work with the keras training mode. There seems to be an issue with the inference of the shape of some outputs, so some other method might have to be added to class 
LinearSVM in utils_TFlearn. 


