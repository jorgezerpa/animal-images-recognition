import matplotlib.pyplot as plt
import numpy as np
import os
import pprint

pp = pprint.PrettyPrinter(indent=4)


# FUNCTION TO CREATE THE DATASET
import joblib
from skimage.io import imread
from skimage.transform import resize


def resize_all(src, pklname, include, width=150, height=None):
    """
    load images from path, resize them and write them as arrays to a dictionary, 
    together with labels and metadata. The dictionary is written to a pickle file 
    named '{pklname}_{width}x{height}px.pkl'.
     
    Parameter
    ---------
    src: str
        path to data
    pklname: str
        path to output file
    width: int
        target width of the image in pixels
    include: set[str]
        set containing str
    """
     
    height = height if height is not None else width
     
    data = dict()
    data['description'] = 'resized ({0}x{1})animal images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []   
     
    pklname = f"{pklname}_{width}x{height}px.pkl"
 
    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            # print(subdir)
            current_path = os.path.join(src, subdir)
 
            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    im = imread(os.path.join(current_path, file))
                    im = resize(im, (width, height)) #[:,:,::-1]
                    data['label'].append(subdir[:-4]) # -4 for because the name of subdir in "[animal]Head" -> we delete "Head"
                    data['filename'].append(file)
                    data['data'].append(im) # im is a numpy array -> [[[]]] -> l1:image, l2:pexels row, l3:pexel
 
        joblib.dump(data, pklname) #store data on file -> relative path



data_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'imageRecognition', 'src', 'datasets', 'Image')) #path to images



    #CREATING THE DATASET
base_name = 'animal_faces'
width = 80
 
include = {'ChickenHead', 'BearHead', 'ElephantHead', 'EagleHead', 'DeerHead', 'MonkeyHead', 'PandaHead'}
 
#Uncomment the next line for call the create dataset function (not needed if it is already created)
# resize_all(src=data_path, pklname=base_name, width=width, include=include)



    # LOAD DATA FRIN DISK AND LOGGING IT
from collections import Counter # to count the sequecy of element in an iterable -> how many time each one repeat

data = joblib.load(f"{base_name}_{width}x{width}px.pkl") #name of created file

print('number of samples: ', len(data['data']))
print('keys: ', list(data.keys()))
print('description: ', data['description'])
print('image shape: ', data['data'][0].shape)
print('labels:', np.unique(data['label']))
 
print(Counter(data['label']))















