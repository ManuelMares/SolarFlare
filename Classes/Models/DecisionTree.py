import matplotlib as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from Classes.Models.AbstractPredictionModel import PredictionModel

class DecisionTree(PredictionModel):
    model = None
    def __init__(self):        
        self.model = DecisionTreeClassifier()
        self.name = "Decision Tree"

    