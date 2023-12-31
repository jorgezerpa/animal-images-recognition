import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog

# Note that for compatibility with scikit-learn, the fit and transform methods take both X and y as parameters, even though y is not used here.


class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """
    def __init__(
        self, 
        y=None, 
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(3, 3), 
        block_norm='L2-Hys'
    ):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])