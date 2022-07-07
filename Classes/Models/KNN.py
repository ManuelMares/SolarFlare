import matplotlib as plt
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from Classes.Models.AbstractPredictionModel import PredictionModel

class KNN(PredictionModel):
    model = None
    def __init__(self):        
        self.model = KNeighborsClassifier(n_neighbors = 2)
        self.name = "KNN"

