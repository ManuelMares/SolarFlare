import matplotlib as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from Classes.Models.AbstractPredictionModel import PredictionModel

class DecisionTree(PredictionModel):
    model = None
    
    #criterion          [string]        default = gini (gini, entropy, log_loss)
    #Splitter           [string]        default = best, random
    #max_depth          [int]           defautl = None
    #min_samples_split  [int or float]  default = 2
    #min_samples_leaf   [int or float]  default = 1
    #min_weight_fraction_leaf   [float] default = 0.0
    #max_features       [int, float, string]      default = None (auto, sqrt, log2)
    #random_state       [int]           default = None (random state)
    #max_leaf_nodes     [int]           default = None
    #min_impurity_decrease [float]      default = 0.0
    #class_weight       [dict, list of dict, "balanced"]        default = None
    #ccp_alpha          [float +]       default = 0.0
    
    hyperparameters = [
                        {"criterion": "gini", "max_depth": None, "spliter":"best", "max_leaf_nodes": None, "min_samples_split": 2, "min_samples_leaf": 1, "random_state": 42},
                        {"criterion": "gini", "max_depth": None, "spliter":"best", "max_leaf_nodes": None, "min_samples_split": 2, "min_samples_leaf": 1, "random_state": 42},
                        {"criterion": "gini", "max_depth": None, "spliter":"best", "max_leaf_nodes": None, "min_samples_split": 2, "min_samples_leaf": 1, "random_state": 42},
                        {"criterion": "gini", "max_depth": None, "spliter":"best", "max_leaf_nodes": None, "min_samples_split": 2, "min_samples_leaf": 1, "random_state": 42}
                      ]
    
    def __init__(self):      
        self.Combinations_Set_HyperParameters(self.hyperparameters)        
        self.model = DecisionTreeClassifier()
        self.name = "Decision Tree"
        
    def ModelOperation_UpdateParameters(self, parametersIndex):
        self.model = DecisionTreeClassifier( C = self.hyperparameters[parametersIndex]["C"], 
                                             gamma = self.hyperparameters[parametersIndex]["gamma"], 
                                             kernel = self.hyperparameters[parametersIndex]["kernel"] )

    