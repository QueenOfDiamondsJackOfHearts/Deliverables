"""
This file contains useful functions and classes to be used in the constuction, training, and evaluation of the models presented in the Transfer Learning notebook. 
"""
import sys
import numpy as np
import keras
from keras.preprocessing import image
import scipy
import os
from sklearn.metrics import confusion_matrix

from keras.models import Model

from keras.utils import plot_model
from IPython.display import Image
import tensorflow as tf

import matplotlib.pyplot as plt
import pickle

from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np

def visualize_filter(model, layer_name, filters=None, image=None, save_path='stitched_filters.png',
                     n=1, loss_min=0, grad_check=True, steps=20, step_size=1):
    """
    This creates a visualization of the filters in a given layer by using gradient ascent on either a random input
    image, or a selected image. It returns a list of kept filters and save an nxn image of filters stiched together
    and saved to path, save_path. We perform gradient asscent to maximize a loss function which maximizes the 
    output of that node, and forces others to zero.
    
    Input:
    ------------------
    model: a keras model
    
    layer_name: a string identifying a layer in the above model, (layer must have output dimension of 4)
    
    filters: None or some iterable list of indices of filters to visualizes. If none, all features will be 
             visualized (or at least attempted). 
    
    image: if not none, the image to use as a starting point in gradient ascent
            
    save_path: a string, where the image is saved
    
    n: an integer, the square root of how many filters to use in the output image. (will error if not enough 
       kept filters)
    
    loss_min: float, a lower bound on the loss of filter we will keep
    
    grad_check: bool, if true we check if the loss of a filter is 0 on an image, and skip it if true
            
    Output: 
    --------------------
    
    kept_filters: a list of images equal in size to those of the inputs of model.

    
    NOTE: requires channel last format as of current version. adapted from: 
    https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
    
    """
    

    import numpy as np
    import time
    from keras.preprocessing.image import save_img
    from keras.applications import vgg16
    from keras import backend as K
    
    # util function to convert a tensor into a valid image
    def deprocess_image(x):
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + K.epsilon())
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        if K.image_data_format() == 'channels_first':
            x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('float64')
        return x
    
    def l2_norm(x):
        return K.eval(K.sqrt(K.mean(K.square(x))))

    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())
    
    
    #infer image size and input tensor from model
    img_width = int(model.input.shape[1])
    img_height = int(model.input.shape[2])
    input_img = model.input
    
    
    if image: #... process and load image
        size=(img_width,img_height)
        im=img.open(image)
        im=im.resize(size,img.ANTIALIAS)
        im.load()
        img_data=np.asarray(im, dtype="float64" )
        img_data=np.expand_dims(img_data,axis=0)
    
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    
    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    
    if not filters:# if default to none or empty list, creates an iterable to loop over all layers
        filters=range(int(layer_output.shape[3]))
    
    kept_filters = []
    for output_index in filters:
        
        print('Processing filter %d' % output_index)
        start_time = time.time()
        
        loss = K.mean(layer_output[:,:,:,output_index])

        #we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent



        if image: #... use image
            input_img_data=img_data
        else: #... use a random array image
            input_img_data = np.random.random((1, img_width, img_height, 3))
            #transform the data to have values in [0,255]
            input_img_data = (input_img_data - 0.5) * 20 + 128



        for i in range(steps):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step_size

            print('Current loss value:', loss_value)
            if loss_value <= 0:
                # some filters get stuck to 0, we can skip them
                break

        # decode the resulting input image
        if loss_value > loss_min:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))

            
        end_time = time.time()
        print('Filter %d processed in %ds' % (output_index, end_time - start_time))


    #we stich the best n^2 filters on a n x n grid.
    #---------------------------------------------------------------


    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top n^2 filters in the image.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for
    # our n x n filters of size equal to the input dimension of our model, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                             (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    # save the result to disk
    save_img(save_path,stitched_filters)        
    
#Other functions useful for setting up the data set
def get_file_leaves(Dir):

    """
    returns all files reachable from a directory in a list containing a path from that directory
    
    Input:
    --------------------
    Dir: a string of the path to the directory that is the root of the file tree
    
    
    Outputs:
    --------------------
    files: a list of strings representing file paths which can be reached from DIr
    
    --------------------
    NOTE: requires import of os
    """
    from shutil import copy2
    #first we use walk to gather the files...
    tree=os.walk(Dir)
    files=[]
    for x in tree:
        for f in x[2]:
            path=x[0]+'/'+f
            files.append(path)
        
    return files
    

def copy2_new_dir(file_paths, new_dir):
    """
    returns all files reachable from a directory in a list containing a path from that directory
    
    Input:
    --------------------
    file_paths: a list of strings representing file paths to be copied to new dir
    
    
    Outputs:
    --------------------
    None
    
    --------------------
    NOTE: requires copy2 from shutils
    """
    from shutil import copy2
    #now we make a new directory and copy the files
    for path in file_paths:
        copy2(path,new_dir)

def sort_test_train(file_paths, class_name, test_percent=.1):
    """
    this function takes a list of file paths which all lead to image files and randomly sorts them based on 
    the test percentage then assigns them to a "test/class_name" directory or "training/class_name" directory
    
    It is assumed that these directories are reachable from the current working directory
    
    Input:
    --------------------
    file_paths: a list of strings representing file paths to be copied to new dir
    
    class_name: class name to be inferred 
    
    test_percent: a value in (0,1) to denote how much of the data should be sent to the test directory
    
    Outputs:
    --------------------
    None
    
    --------------------
    NOTE: Requires numpy 
    """
    
    from numpy.random import shuffle  
    import numpy as np
    
    n=len(file_paths)
    cutoff= int(np.ceil(n*test_percent)) # this gives a delimiter for the file
    
    shuffle(file_paths)
    
    test=file_paths[0:cutoff]
    training=file_paths[cutoff:]
    
    copy2_new_dir(test, 'test/'+class_name)
    copy2_new_dir(training, 'training/'+class_name)
    
def test_model(data,lables,model,give_confidence=True):
    """
    Tests a trained binary classification model on a test set of data, with a corresponding list of lables. 
    The model makes a prediction on the data, and records which images are misclassified and as what, and returns
    a this data as lists. 
    
    
    
    Input:
    ------------------
    model: a trained keras binary classification model
    
    data: a list preprocessed numpy arrays which are valid inputs for the model
    
    lables: a list the same lenght as data such that the ith item in data had the same label as the ith element in 
            lables. This list has values 0 or 1
            
    give_confidence: if this is true, the model also returns the confidence of the classifcation.
            
    Output: (pred_1_but_0,pred_0_but_1,conf)
    --------------------
    
    MisClassified_0: list of indices whose classification was labled 1, but the expected output was 0
     
    MisClassified_1: "     "     "       "     "        "      "     0    "    "        "       "   1
    
    conf: an array of 
    
    """
    MisClassified_0=[]
    MisClassified_1=[]
    
    
    for i, arr in enumerate(data):
        arr=np.expand_dims(arr,0)
        prediction=model.predict_classes(arr)
        if not prediction==lables[i]:
            if lables[i]==0:
                MisClassified_0.append(i)
            else:
                MisClassified_1.append(i)
    if give_confidence:
        conf=model.predict(data)
        return MisClassified_0, MisClassified_1, conf
    else:
        return MisClassified_0, MisClassified_1
    
def dict_to_list(dict):
    """
    this function takes a dictionary, dict, whose keys have distinct integers, spanning [0..len-1], and creates a list 
    ordering the keys by these integer values.

    NOTE: This is particularly useful in creating readible predictions while using keras, which returns class indices
    as a dictionary whose keys are classes, and whose values are their indices
    """
    
    keys=dict.keys()
    n=len(keys)
    Ls= list(range(0,n))

    for k in keys:
        Ls[dict[k]]=k

    return Ls
                
def bin_classify_from_path(model, img_path, classes, tol=10**-10):
    """
    Outputs the result of running an img found at img_path through the binary classifier, model.
    The output will be a string, representing the class that is selecte, or a third option "Not Certain" if the
    output of the model is not able to classify the image based on the tolerance, tol. 
    
    The output classes are derived from classes, which is a list of the strings of class names with the index [0]
    reffering to the low activation class, and index [1] reffering to the high activation class
    
    Model assumed to have the input data of shape (Batch, length, height, channels), adjust model accordingly
    
    
    NOTE: model must be binary, and constructed in Keras, and this function requires 
    numpy
    keras.preprocessing.image
    """
    
    input_size=(model.get_input_shape_at(0)[1],model.get_input_shape_at(0)[2]) 
    img=image.load_img(img_path,target_size=input_size)
    img=image.img_to_array(img)
    
    #add another dimension as imput is expected as an array of images
    img=np.expand_dims(img,axis=0) 
    
    result=model.predict(img)
    if result[0][0]<tol:
        return classes[0]
    elif result[0][0]>1-tol:
        return classes[1]
    else:
        print("Not classified! Certainty: "+repr(result[0][0]))
        return "Not classified"
    
def bin_classify_gen (model, image_set, String_classes=False,tol=10**-10):
    """
    runs a binary classifier model, constructed in keras, on an image set represented by a keras imagegenerator
    and returns the list of predictions in string form or index form(default)
    
    Input:
    ------------------
    model: a trained keras binary classification model
    
    image_set: a keras image generator object whose lables have the same meaning as those of the model. (i.e if 1 is 
               'dog' in the model, the same )
    
    String_classes: a boolean value to determine if the result should be return in its encoding as an integer, or if it 
                  should be interpreted as a string. By default it is false
    
    tol: a float value to represent the certainty with which we can consider a prediction a classification
            
    Output:
    --------------------
    results: the list of model outputs, either class strings (not yet supported) or confidences
    """
    #ensures that the model size and image set agree
    input_size=(model.get_input_shape_at(0)[1],model.get_input_shape_at(0)[2]) 
    image_set.target_size=input_size
    
    
    
    
    results=model.predict_generator(image_set)
    #TODO COMPLETE string fucntionality
    if String_classes:
        cls_ind=image_set.class_indices #dictionary of form ('class', index)
        classes=dict_to_list(cls_ind)
       
        for result in results:
            if result[0]<tol:
                     classes[0]
            elif result[0]>1-tol:
                return classes[1]
            else:
                print("Not classified! Certainty: "+repr(result[0][0]))
                return "Not classified"
    else:
        return results
    
def save_dict(filepath,dic):
    """
    saves a dictionary, dic to the specified filepath a a pickled binary file 
    
    Input:
    ------------------
    filepath: string representing the filepath
    
    dic: dictionary to be saved
            
    Output:
    --------------------
    None 
    """
    import pickle
    with open(filepath, 'wb') as file_pi:
        pickle.dump(dic, file_pi)
        
def plot_confusion_matrix(model,data,labels,classes,title='Confusion Matrix', cmap=None ):
    """
    plots a confusion matrix of a keras classification model, for given data with associated lables.
    NOTE: the below code has only been verified on a binary classifier
    
    Input:
    ------------------
    model: keras model object used for classification
    
    data: list of data arrays which have valid input shape for the keras model
    
    lables: list of numeric lables which correspond in order to the arrays in data
    
    classes: a list of strings of labels whose indices correspond to the numeric lables in lables
    
    cmap: a color map as defined in pyplot, when None defaults to blue 
            
    Output: 
    --------------------
    cm: a ndarray representing the confusion matrix
    
    code adapted from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import numpy as np
    
    predicts=model.predict(data) #class predicitions
    if cmap == None: 
        cmap=plt.cm.Blues
        
    cm=confusion_matrix(labels,predicts) #create confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #      plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    return cm

def test_model_svm(data,lables,model,give_confidence=False):
    """
    Tests a trained binary classification model on a test set of data, with a corresponding list oflables. 
    The model makes a prediction on the data, and records which images are misclassified and as what, and 
    returns this data as lists. 
    
    Input:
    ------------------
    model: an sklearn svm
    
    data: a list preprocessed numpy arrays which are valid inputs for the model
    
    lables: a list the same lenght as data such that the ith item in data had the same label as the ith 
    element in lables. This list has values 0 or 1
            
    give_confidence: if this is true, the model also returns the confidence of the classifcation.
            
    Output: (pred_1_but_0,pred_0_but_1,conf)
    --------------------
    
    MisClassified_0: list of indices whose classification was labled 1, but the expected output was 0
     
    MisClassified_1: "     "     "       "     "        "      "     0    "    "        "       "   1
    
    conf: an array of 
    
    """
    MisClassified_0=[]
    MisClassified_1=[]
    
    
    for i, arr in enumerate(data):
        arr=np.expand_dims(arr,0)
        prediction=model.predict(arr)
        if not prediction==lables[i]:
            if lables[i]==0: #if the expected label is 0
                MisClassified_0.append(i)
            else:
                MisClassified_1.append(i)
    if give_confidence:
        conf=model.predict(data)
        return MisClassified_0, MisClassified_1, conf
    else:
        return MisClassified_0, MisClassified_1

def display_NParray(arr):
    """
    displays an array as an RGB array and displays the image
    
    Input:
    ------------------
    arr: an array which can be interpreted as an image
            
    Output:
    --------------------
    """
    plt.imshow(arr, interpolation='nearest')
    plt.show()


class LinearSVM(Layer):
    """
    This layer takes a 2D Tensor (batch,features), and outputs the results of a SVM applied to that Tensor, returning 
    a 1D tensors of predictions of size (batch).
    
    The SVM to be used is a LinearSVM implemented in sklearn, or an svm represented with an n-dimensional vector, and 
    scalar bias. 
    """
    def __init__(self, coef, intercept, **kwargs):
        self.coef=coef
        self.intercept=intercept
        self.use_bias=False
        self.trainable=False
        super(LinearSVM, self).__init__(**kwargs)
        

        
    def build(self, input_shape):
        """
        We expect an input shape of the form (batch,row,column,channel)
        TODO: allow for channel-first functionality 
        
        if we have our constant parameter set to true, then we shall make our kernel a normalized 
        """
        #for some reason the config file with save the np_arrays as dictionaries of the form
        #{"type":ndarray, "value":<actual array> } so this code allow the model to load from configure
        if isinstance(self.coef,dict): 
            self.coef=self.coef['value']
        if isinstance(self.intercept,dict):  
            self.intercept=self.intercept['value']
            
        self.kernel=K.variable(self.coef)
        self.kernel=K.transpose(self.kernel)
        self.intercept_const=K.variable(self.intercept) 
        super(LinearSVM, self).build(input_shape)  # Be sure to call this at the end
        
    def call(self, x):
        
        return (K.dot(x, self.kernel)+self.intercept_const)
    
    def get_config(self):
        config={"coef":self.coef,"intercept":self.intercept}
        base_config = super(LinearSVM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        

    def compute_output_shape(self, input_shape):
        return (input_shape[0],1)