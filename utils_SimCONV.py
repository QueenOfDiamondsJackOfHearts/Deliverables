"""
This file contains useful functions and classes to be used in the constuction, training, and evaluation of the models presented in the "First Model Visualizations" notebooks. 
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
    
    predicts=model.predict_classes(data) #class predicitions
    pred_confidence=model.predict(data)  #numeric confidence
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