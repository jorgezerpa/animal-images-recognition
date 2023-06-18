import os
import joblib
import numpy as np
from resize_all import resize_all

    #CREATING THE DATASET
data_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'imageRecognition', 'src', 'datasets', 'Image')) #path to images
base_name = 'animal_faces'
width = 80 
height = 80
include = {'ChickenHead', 'BearHead', 'ElephantHead', 'EagleHead', 'DeerHead', 'MonkeyHead', 'PandaHead'}
###Uncomment the next line for call the create dataset function (not needed if it is already created)
# resize_all(src=data_path, pklname=base_name, width=width, include=include)



    # LOAD DATA FROM DISK AND LOGGING IT
from collections import Counter # to count the sequecy of element in an iterable -> how many time each one repeat

data = joblib.load(f"{base_name}_{width}x{height}px.pkl") #name of created file

###Uncoment to log data features
# print('number of samples: ', len(data['data']))
# print('keys: ', list(data.keys()))
# print('description: ', data['description'])
# print('image shape: ', data['data'][0].shape)
# print('labels:', np.unique(data['label'])) 
# print(Counter(data['label']))


    #PREPATING TRAIN AND TEST DATA
from sklearn.model_selection import train_test_split
 
X = np.array(data['data'])
y = np.array(data['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True,random_state=42)

    #PROCESSING AND TRANSFORMERS
from sklearn.preprocessing import StandardScaler, Normalizer

from transformers.HogTransformer import HogTransformer
from transformers.rgbToGrayTransformer import RGB2GrayTransformer

grayify = RGB2GrayTransformer()
hogify = HogTransformer(pixels_per_cell=(14, 14), cells_per_block=(2,2), orientations=9, block_norm='L2-Hys')
scalify = StandardScaler()
 
# call fit_transform on each transform converting X_train step by step
X_train_gray = grayify.fit_transform(X_train)
X_train_hog = hogify.fit_transform(X_train_gray)
X_train_prepared = scalify.fit_transform(X_train_hog)
 
print(X_train_prepared.shape)







