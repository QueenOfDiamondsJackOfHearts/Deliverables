# Deliverables

## Contents 

### Notebooks
For any of these notebooks, if you have a question on a function's implementation and said function is not in the notebook, 
check the associated utility file for docs and source code, or in notebook type "\<function name\>?" to simply pull up docs. 

* "First model evaluation and visualization"
    contains training, testing, and vizualization script for the simple convolutional model, including confusion matrix,
    showing misclassified images, and convolutional filter visualization
    
    Associated utility file: utils_SimCONV.py

* "Transfer Learning" 
    contains training, testing, and vizualization script for the Transfer Learning model, including confusion matrix,
    showing misclassified images, and convolutional filter visualization. Also contains an exploration of different SVM 
    kernels, and script for the pipeline of data => feature extractor => classifier
    
    Associated utility file: utils_TFLearn.py
    
* "Google Custom Search API"(found in data gathering) 
    contains information on how we constructed our dataset through use of google custom search. Provides a basic script which 
    can be used to recreate  the data set used to train and evaluate the model. To use you must aquire and API key and set up 
    a custom search engine through the google api console. (see step 4 below)
    
    Associated utility file: utils_TFLearn.py
### Models
Model files are provide, which are prediction ready. This include some version of the simple convolutional model and the VGG16 transfer learning model with linear SVM

Found in models_LFS, located in google drive

### Data Set
Data set used to train and evaluate models

found in Data_Set1, located in google drive
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

You should always run the first block, which imports the packages needed. You should also run any function definitions you see. Other than that, the general rule of thumb is that to run a given cell, you must run the one above. This is not always the case as some tasks, like loading a pre-trained model from "model_LFS" has no real requirements except importing the necessary modules. Another important point is that for the simple convolutional model, while it will train under the current conditions in the notebook, training in the notebook is generally illadvised as memory issues can become a concern. It is thus advisable to train the models in a separate script.

### Step 4. Exploring the Data Gathering folder

To use this notebook, you must set your own Custom Search Engine. A tutorial can be found: here https://developers.google.com/custom-search/docs/tutorial/creatingcse. This will involve both aquiring an API key, and creating an actual search engine. There is a relatively simple console that can be used to make search engines quickly, with the ability to create an XML file to get a particularly fine tuned search engine. In the current work, I have not explored this functionality in depth, but it may be interesting. Check the Utility module (GCS_utils) for details on the functions used to create the dataset, as well as the functions used for file manipulations.

## Notes:

These models are prediction ready, but the transfer learning model does not quite work with the keras training mode. There seems to be an issue with the inference of the shape of some outputs, so some other method might have to be added to class 
LinearSVM in utils_TFlearn. 


