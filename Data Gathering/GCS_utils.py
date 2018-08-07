from apiclient.discovery import build

from googleapiclient.discovery import build
import pprint

import os
import requests
import sys

def img_search(search_term, api_key, cse_id, num, **kwargs):
    """
    Conducts an image search of some query given a custom search engine. If no items found, None returned
    and prints that there are no results
    
    NOTE: The search engine must have images enabled
    
    Input:
    --------------------
    search_term: a string put into the query
    
    api_key: a string of the api key used to access the data
    
    cse_id: a string which identifies the search engine to use
    
    num_img: the number of top images which should be retrieved
    
    kwargs: optional arguments used for querying 
    
    Outputs:
    --------------------
    res['items']: a JSON of metadata of the returned search results from the JSON file. It is interpreted in python
                  as a list of dictionaries
    --------------------
    NOTE: requires google-api-python-client, your own API key, and a CSE
    """
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, searchType='image',num=num,**kwargs).execute()
    
    try:
        res["items"] 
    except KeyError:
        print("There are no items returned for this search")
        return 
    return res["items"] #items returns relevant metadata about the sites returned

def save_img(url, folder, name):
    """
    downloads an image from a url to a specified folder, with a specified name
    
    Input:
    --------------------
    url: string http address of the image 
    
    folder: a string of the path from the working directory to the desired save location
    
    name: a string postpended with the .jpg tag
    
    Outputs:
    --------------------
    None, but image is saves
    --------------------
    NOTE: requires import of module request
    """
    cwd=os.getcwd()
    try:
        os.chdir(folder)
    except FileNotFoundError:
        os.makedirs(folder)
        os.chdir(folder)
        
    try:
        img_data=requests.get(url).content
        with open(name, 'wb') as handler:
            handler.write(img_data)
    except:
        os.chdir(cwd)
        print('Error occured: ', sys.exc_info())
    os.chdir(cwd)

def parse_fname(url):
    """
    This function should take a url linking to an image file of form '.jpg','.jpeg','.png','.gif' , and parse out the 
    name. Here the name is defined as the string from the last '/' before the suffix to the image. If no suffix 
    is found, None is returned and user is prompted to verify that the link ends in an image. 
    
    TODO: add other formats, 
    
    NOTE: no suffix should not appear before its use as the postpend of the image file, otherwise issuses could occur
    Also assumes that only one suffix is present in a url
    
    Input:
    --------------------
    url: string http address of the image 
 
    Outputs:
    --------------------
    name: string that is the given name of the image, or None as described above
    
    --------------------
    """
    
    accepted_formats=['.jpg','.jpeg','.png','.gif']
    i=0
    for form in accepted_formats:
        ind=url.find(form)
        if ind>-1: #if a given format is found...
            end=ind
            suff=form
            break  #we record that and break the for loop
        i=i+1
    if i==len(accepted_formats):
        print('------------------------------------')
        print('no accepted format found, ensure file links to image')
        print(url)
        return None
#     try: 
#         end=url.index('.jpg')
#     except ValueError:
#         print('------------------------------------')
#         print('.jpg not found, ensure file links to image')
#         print(url)
#         return url
    i=end-1  
    while i>-1:
        if url[i]=='/':
            start=i+1
            name=url[start:end]+suff
            return name
        i=i-1
    print('no / found, the whole url is the name')
    return url[:end].jpg

#__________________________________________________________________
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
            