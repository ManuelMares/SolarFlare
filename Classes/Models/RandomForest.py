import matplotlib as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from Classes.Models.AbstractPredictionModel import PredictionModel

class RandomForest(PredictionModel):
    model = None
    #n_estimators       [int]           default = 100
        #more trees, better generalization, but high cost
        #1,2,5,10,20,50,100
    #max_depth          [int]           default = 100 
        #longest path between root node and leaf node
        #This parameter causes great overfitting of the tree
        #But it is compensated in the forest
        #1, 2, 3, 5, 10, 15, 20, 30
    #max_leaf_nodes     [int]           default = None
        #restricts the growth of the tree by condition the splitting
        #too low, underfit, usually overfits beyond 25
        #2,5,10,20,50
    #min_samples_split  [int or float]  default = 2     
        #minimum number of observation in any node before splitting
        #value 2 causes pure node, hence, overfitting.
        #increasing reduces the number of splits and overfitting
        #too high, underfits
        #2, 5, 10, 50, 100, 1000
    #min_samples_leaf   [int or float]  default = 1
        #minimum number of samples in leaf after splitting
        #prevents overfitting
        #10, 50, 100, 300, 500, 800, 1000
    #max_samples        [int or float]  default = None
        #what fraction of the original daataset is given to each tree
        #.001, .005, .01, .05, .1, .2, .5, 1    
    #max_features       [int or float]  default = sqrt (log2, None, sqrt)
        #features provided to each tree
        #best around 6
        #1, 2, 3, 4, 5,, 6, 7, 8, 9, 10


#Other parameters
    #criterion          [string]        default = gini (gini, entropy, log_loss) | quality of split
    #min_weight_fraction_leaf [float]   default = 0.0
    #min_impurity_decrease [float]      default = 0.0
    #bootstrap          [bool]          default = True
    #oob_score          [bool]          default = False
    #n_jobs             [int]           default = None
    #random_state       [int]           default = None (RandomState instance, None)
    #verbose            [int]           default = 0
    #warm_start         [bool]          default = False
    #class_weight       [dict or list of dict]    default = None ("balanced", "balanced_subsample")
    #ccp_alpha          [non-negative flaot]      default = 0.0


    hyperparameters = [
                        {"n_estimators": 100, "max_depth": 100, "max_leaf_nodes": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_samples":None, "max_features": "sqrt", "random_state": 42},
                        {"n_estimators": 100, "max_depth": 100, "max_leaf_nodes": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_samples":None, "max_features": "sqrt", "random_state": 42},
                        {"n_estimators": 100, "max_depth": 100, "max_leaf_nodes": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_samples":None, "max_features": "sqrt", "random_state": 42},
                        {"n_estimators": 100, "max_depth": 100, "max_leaf_nodes": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_samples":None, "max_features": "sqrt", "random_state": 42}
                      ]
    
    def __init__(self):        
        self.Combinations_Set_HyperParameters(self.hyperparameters)
        self.model = RandomForestClassifier()
        self.name = "Random Forest"
    
    def ModelOperation_UpdateParameters(self, parametersIndex):
        self.model = RandomForestClassifier( C = self.hyperparameters[parametersIndex]["C"], 
                                             gamma = self.hyperparameters[parametersIndex]["gamma"], 
                                             kernel = self.hyperparameters[parametersIndex]["kernel"] )


