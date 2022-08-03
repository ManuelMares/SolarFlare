from inspect import Parameter
import matplotlib as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import ParameterSampler
from Classes.Models.AbstractPredictionModel import PredictionModel

class SVC(PredictionModel):
    model = None

    #C                  [float +]       default = 1.0 |regularization parameter
            #penalty parameter of the error. Control smoothnes of decision boundary
            #best in order 0.1, 1, 10, 100, 1000   
    #kernel             [string]        default = rbf (linear, poly, rbf, sigmoid, precomputed or callable)
            #rbf, sigmoid poly
    #degree             [int]           default = 3
            #only for poly kernel, try 1,2,3,4,5
    #gamma              [string or float]         default = scale (scale, auto, or number)  |Kernel coeficient for rbf or poly and sigmoid
            #0.001, 0.002, 0.003, .005, 0.01, 0.1

#Other parameters
    #coef0              [float]         default = 0.0
    #shrinking          [bool]          default = True
    #probability        [bool]          default = False
    #tol                [float]         default = 1e-3
    #cache_size         [float]         default = 200   |In MB
    #class_weight       [dict or "balanced"]      default = None
    #verbose            [bool]          default = False
    #max_iter           [int]           default = -1
    #decision_function_shape   [string] default = ovr (ovo, ovr)
    #break_ties         [bool]          default = False
    #random_state       [int]           default = None


    hyperparameters = [
                        {"C": 0.2, "gamma": 0.1, "kernel": "rbf", "degree": 3, "random_state": 42},
                        {"C": 0.2, "gamma": 0.1, "kernel": "rbf", "degree": 3, "random_state": 42},
                        {"C": 0.3, "gamma": 0.1, "kernel": "rbf", "degree": 3, "random_state": 42},
                        {"C": 0.4, "gamma": 0.1, "kernel": "rbf", "degree": 3, "random_state": 42}
                      ]
    
    def __init__(self):
        self.Combinations_Set_HyperParameters(self.hyperparameters)
        self.model = svm.SVC()
        self.name = "SVM"
    
    def ModelOperation_UpdateParameters(self, parametersIndex):
        self.model = svm.SVC( C = self.hyperparameters[parametersIndex]["C"], 
                              gamma = self.hyperparameters[parametersIndex]["gamma"], 
                              kernel = self.hyperparameters[parametersIndex]["kernel"] )


