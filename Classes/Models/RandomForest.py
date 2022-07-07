import matplotlib as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from Classes.Models.AbstractPredictionModel import PredictionModel

class RandomForest(PredictionModel):
    model = None
    
    def __init__(self):        
        self.model = RandomForestClassifier()
        self.name = "Random Forest"
