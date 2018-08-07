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
            