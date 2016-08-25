from sklearn import datasets
from sklearn import preprocessing
import numpy as np

# 1. Getting sample data from external sources
boston = datasets.load_boston() #return bunch:dictionary-like object
print boston.DESCR
boston_X, boston_y = boston.data, boston.target

# 2. Creating sample data for toy analysis
# X, y, coef = datasets.make_regression(coef=True)
# #100 samples x 100 features; 10 features are responsible for the target data generation
# complex_reg_data = datasets.make_regression(1000,10,5,2,1.0)
# 1000 samples x 10 features, 5 contributed features, 2 targets, bias factor 1.0

# 3. Scaling data to the standard normal
boston_X_scaled = preprocessing.normalize(boston_X[:, :3])
#.scale(), .StandardScalar(), .MinMaxScalar()
print boston_X_scaled.mean(axis=0)
print boston_X_scaled.std(axis=0) #[1. 1. 1.]

# 4. Creating binary features through thresholding
boston_y_new = preprocessing.binarize(boston.y, threshold = boston.target.mean())



