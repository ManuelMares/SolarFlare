import matplotlib as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from Classes.Models.AbstractPredictionModel import PredictionModel

class NB_Gaussian(PredictionModel):
    model = None

    #alpha              [float]         default = 1.0 
    #fit_prior          [string]        default = True
    #class_prior        [array-like of shape (n_classes, )]  default = None
    #min_categories     [int or array-like of shape (n_features, )]     default = None

    hyperparameters = [
                        {"alpha": 1.0},
                        {"alpha": 1.0},
                        {"alpha": 1.0},
                        {"alpha": 1.0}
                      ]
    
    model = None
    def __init__(self):      
        #Create a new model for 
        #ComplementNB
        #categoricalNB
        self.Combinations_Set_HyperParameters(self.hyperparameters)  
        self.model = GaussianNB()
        self.name = "Naive Bayes"
        
    def ModelOperation_UpdateParameters(self, parametersIndex):
        self.model = GaussianNB( C = self.hyperparameters[parametersIndex]["C"], 
                                 gamma = self.hyperparameters[parametersIndex]["gamma"], 
                                 kernel = self.hyperparameters[parametersIndex]["kernel"] )
