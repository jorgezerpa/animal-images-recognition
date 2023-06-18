import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import skimage
 
# Note that for compatibility with scikit-learn, the fit and transform methods take both X and y as parameters, even though y is not used here.

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])
     
 
