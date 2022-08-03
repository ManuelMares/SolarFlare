import matplotlib as plt
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from Classes.Models.AbstractPredictionModel import PredictionModel

class KNN(PredictionModel):
    model = None

    #n_neighbors        [int]           default = 5
    #weights            [string]        default = uniform (uniform, distance, *callable) | callable is an user defined function
    #algorithm          [string]        default = auto (auto, ball_tree, kd_tree, brute) 
    #leaf_size          [int]           default = 30     
    #p                  [int]           default = 2
                        #minkowski parameter, p = 1 = manhattan, p = 2 = euclidean
    #metric             [string]        default = minkowski (euclidean, manhattan, chebyshev, minkowski(p, w), seuclidean(V), mahalanobis(V or VI))
                        #called as   
                        #from sklearn.metrics import DistanceMetric
                        #DistanceMetric.get_metric('euclidean')
    #metric_params      [dict]          default = None |additional keyword arguments for the metric function
                        # minkowski(p, w)
                        # seuclidean(V)
                        # mahalanobis(V or VI)
    #n)jobs             [int]           default = 1 | -1 means use all processors

    hyperparameters = [
                        {"n_neighbors": 5,  "leaf_size": 30, "metric": "metric", "algorithm": "auto"},
                        {"n_neighbors": 8,  "leaf_size": 30, "metric": "metric", "algorithm": "auto"},
                        {"n_neighbors": 10, "leaf_size": 30, "metric": "metric", "algorithm": "auto"},
                        {"n_neighbors": 15, "leaf_size": 30, "metric": "metric", "algorithm": "auto"}
                      ]
    
    def __init__(self):      
        self.Combinations_Set_HyperParameters(self.hyperparameters)         
        self.model = KNeighborsClassifier(n_neighbors = 2)
        self.name = "KNN"
        
    def ModelOperation_UpdateParameters(self, parametersIndex):
        self.model = KNeighborsClassifier( C = self.hyperparameters[parametersIndex]["C"], 
                                           gamma = self.hyperparameters[parametersIndex]["gamma"], 
                                           kernel = self.hyperparameters[parametersIndex]["kernel"] )


